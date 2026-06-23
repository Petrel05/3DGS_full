from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .io_utils import Frame


@dataclass
class Observation:
    image_id: int
    xy: np.ndarray


@dataclass
class Reconstruction:
    camera_matrix: np.ndarray
    rotations: list[np.ndarray]
    translations: list[np.ndarray]
    points: np.ndarray
    colors: np.ndarray
    tracks: list[list[Observation]]
    reprojection_rmse: float
    pair_inliers: list[int]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(self, x: tuple[int, int]) -> tuple[int, int]:
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def detect_features(frames: list[Frame], max_features: int = 2500):
    detector = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.015)
    keypoints = []
    descriptors = []
    for frame in frames:
        kps, desc = detector.detectAndCompute(frame.image, frame.mask)
        keypoints.append(kps)
        descriptors.append(desc)
    return keypoints, descriptors


def match_descriptors(desc1: np.ndarray | None, desc2: np.ndarray | None, ratio: float = 0.75):
    if desc1 is None or desc2 is None or len(desc1) < 8 or len(desc2) < 8:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for a_b in raw:
        if len(a_b) != 2:
            continue
        a, b = a_b
        if a.distance < ratio * b.distance:
            good.append(a)
    return good


def estimate_chained_poses(keypoints, descriptors, k: np.ndarray):
    rotations = [np.eye(3, dtype=np.float64)]
    translations = [np.zeros(3, dtype=np.float64)]
    pair_matches = []
    pair_inliers = []

    for i in range(1, len(keypoints)):
        matches = match_descriptors(descriptors[i - 1], descriptors[i])
        pair_matches.append(matches)
        if len(matches) < 12:
            rotations.append(rotations[-1].copy())
            translations.append(translations[-1] + np.array([0.2, 0.0, 0.0]))
            pair_inliers.append(0)
            continue

        pts1 = np.float64([keypoints[i - 1][m.queryIdx].pt for m in matches])
        pts2 = np.float64([keypoints[i][m.trainIdx].pt for m in matches])
        essential, mask = cv2.findEssentialMat(pts1, pts2, k, method=cv2.RANSAC, prob=0.999, threshold=1.5)
        if essential is None:
            rotations.append(rotations[-1].copy())
            translations.append(translations[-1] + np.array([0.2, 0.0, 0.0]))
            pair_inliers.append(0)
            continue
        _, rel_r, rel_t, pose_mask = cv2.recoverPose(essential, pts1, pts2, k)
        rel_t = rel_t.reshape(3)
        new_r = rel_r @ rotations[-1]
        new_t = rel_r @ translations[-1] + rel_t
        rotations.append(new_r)
        translations.append(new_t)
        pair_inliers.append(int((pose_mask > 0).sum()))
    return rotations, translations, pair_matches, pair_inliers


def projection_matrix(k: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return k @ np.hstack([r, t.reshape(3, 1)])


def project_points(points: np.ndarray, k: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    cam = (r @ points.T).T + t.reshape(1, 3)
    z = np.maximum(cam[:, 2:3], 1e-8)
    pix = (k @ cam.T).T
    return pix[:, :2] / z


def build_tracks(keypoints, pair_matches):
    uf = UnionFind()
    for i, matches in enumerate(pair_matches, start=1):
        for match in matches:
            uf.union((i - 1, match.queryIdx), (i, match.trainIdx))

    grouped: dict[tuple[int, int], list[Observation]] = {}
    seen: set[tuple[int, int]] = set()
    for image_id, kps in enumerate(keypoints):
        for kp_id, kp in enumerate(kps):
            node = (image_id, kp_id)
            root = uf.find(node)
            if node in seen:
                continue
            grouped.setdefault(root, []).append(Observation(image_id=image_id, xy=np.array(kp.pt, dtype=np.float64)))
            seen.add(node)

    tracks = []
    for obs in grouped.values():
        by_image = {}
        for item in obs:
            by_image.setdefault(item.image_id, item)
        track = [by_image[i] for i in sorted(by_image)]
        if len(track) >= 2:
            tracks.append(track)
    return tracks


def triangulate_track(track: list[Observation], k: np.ndarray, rotations, translations):
    first = track[0]
    last = track[-1]
    if first.image_id == last.image_id:
        return None
    p1 = projection_matrix(k, rotations[first.image_id], translations[first.image_id])
    p2 = projection_matrix(k, rotations[last.image_id], translations[last.image_id])
    x1 = first.xy.reshape(2, 1)
    x2 = last.xy.reshape(2, 1)
    x_h = cv2.triangulatePoints(p1, p2, x1, x2)
    if abs(x_h[3, 0]) < 1e-9:
        return None
    x = (x_h[:3, 0] / x_h[3, 0]).astype(np.float64)
    if not np.all(np.isfinite(x)):
        return None
    for obs in (first, last):
        depth = (rotations[obs.image_id] @ x + translations[obs.image_id])[2]
        if depth <= 1e-5:
            return None
    return x


def track_rmse(point: np.ndarray, track: list[Observation], k, rotations, translations) -> float:
    residuals = []
    point2 = point.reshape(1, 3)
    for obs in track:
        xy = project_points(point2, k, rotations[obs.image_id], translations[obs.image_id])[0]
        residuals.append(np.linalg.norm(xy - obs.xy))
    if not residuals:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(residuals))))


def sample_color(frame: Frame, xy: np.ndarray) -> np.ndarray:
    h, w = frame.image.shape[:2]
    x = int(round(np.clip(xy[0], 0, w - 1)))
    y = int(round(np.clip(xy[1], 0, h - 1)))
    return frame.image[y, x].astype(np.float64)


def reconstruct(frames: list[Frame], k: np.ndarray, max_features: int = 2500, max_reprojection: float = 8.0) -> Reconstruction:
    keypoints, descriptors = detect_features(frames, max_features=max_features)
    rotations, translations, pair_matches, pair_inliers = estimate_chained_poses(keypoints, descriptors, k)
    tracks = build_tracks(keypoints, pair_matches)

    points = []
    colors = []
    kept_tracks = []
    for track in tracks:
        point = triangulate_track(track, k, rotations, translations)
        if point is None:
            continue
        rmse = track_rmse(point, track, k, rotations, translations)
        if rmse > max_reprojection:
            continue
        points.append(point)
        kept_tracks.append(track)
        colors.append(sample_color(frames[track[0].image_id], track[0].xy))

    if points:
        points_arr = np.vstack(points)
        colors_arr = np.vstack(colors)
        all_err = [track_rmse(p, tr, k, rotations, translations) for p, tr in zip(points_arr, kept_tracks)]
        rmse = float(np.sqrt(np.mean(np.square(all_err))))
    else:
        points_arr = np.zeros((0, 3), dtype=np.float64)
        colors_arr = np.zeros((0, 3), dtype=np.float64)
        rmse = float("inf")

    return Reconstruction(
        camera_matrix=k,
        rotations=rotations,
        translations=translations,
        points=points_arr,
        colors=colors_arr,
        tracks=kept_tracks,
        reprojection_rmse=rmse,
        pair_inliers=pair_inliers,
    )


def reconstruct_with_known_cameras(
    frames: list[Frame],
    k: np.ndarray,
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    max_features: int = 2500,
    max_reprojection: float = 10.0,
) -> Reconstruction:
    keypoints, descriptors = detect_features(frames, max_features=max_features)
    pair_matches = []
    pair_inliers = []
    for i in range(1, len(keypoints)):
        matches = match_descriptors(descriptors[i - 1], descriptors[i])
        pair_matches.append(matches)
        pair_inliers.append(len(matches))
    tracks = build_tracks(keypoints, pair_matches)

    points = []
    colors = []
    kept_tracks = []
    for track in tracks:
        point = triangulate_track(track, k, rotations, translations)
        if point is None:
            continue
        rmse = track_rmse(point, track, k, rotations, translations)
        if rmse > max_reprojection:
            continue
        points.append(point)
        kept_tracks.append(track)
        colors.append(sample_color(frames[track[0].image_id], track[0].xy))

    if points:
        points_arr = np.vstack(points)
        colors_arr = np.vstack(colors)
        all_err = [track_rmse(p, tr, k, rotations, translations) for p, tr in zip(points_arr, kept_tracks)]
        rmse = float(np.sqrt(np.mean(np.square(all_err))))
    else:
        points_arr = np.zeros((0, 3), dtype=np.float64)
        colors_arr = np.zeros((0, 3), dtype=np.float64)
        rmse = float("inf")

    return Reconstruction(
        camera_matrix=k,
        rotations=rotations,
        translations=translations,
        points=points_arr,
        colors=colors_arr,
        tracks=kept_tracks,
        reprojection_rmse=rmse,
        pair_inliers=pair_inliers,
    )
