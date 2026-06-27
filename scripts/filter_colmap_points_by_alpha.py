from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Camera:
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class ImagePose:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def read_cameras(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        cameras[camera_id] = Camera(parts[1], int(parts[2]), int(parts[3]), np.array(parts[4:], dtype=np.float64))
    return cameras


def read_images(path: Path) -> list[ImagePose]:
    images: list[ImagePose] = []
    lines = path.read_text().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        images.append(
            ImagePose(
                image_id=int(parts[0]),
                qvec=np.array(parts[1:5], dtype=np.float64),
                tvec=np.array(parts[5:8], dtype=np.float64),
                camera_id=int(parts[8]),
                name=parts[9],
            )
        )
        # COLMAP text images use two lines per image. The exported datasets in
        # this project keep the POINTS2D line empty, but skip it when present.
        if index < len(lines) and not lines[index].startswith("#"):
            index += 1
    return images


def read_points(path: Path) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    header: list[str] = []
    lines: list[str] = []
    xyz: list[list[float]] = []
    rgb: list[list[int]] = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            header.append(line)
            continue
        if not line:
            continue
        parts = line.split()
        lines.append(line)
        xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
        rgb.append([int(parts[4]), int(parts[5]), int(parts[6])])
    return header, lines, np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.uint8)


def project(points: np.ndarray, pose: ImagePose, camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if camera.model != "PINHOLE":
        raise ValueError(f"Unsupported camera model: {camera.model}")
    fx, fy, cx, cy = camera.params
    rot = qvec_to_rotmat(pose.qvec)
    cam_points = points @ rot.T + pose.tvec
    z = cam_points[:, 2]
    valid = z > 1e-6
    u = fx * (cam_points[:, 0] / np.maximum(z, 1e-6)) + cx
    v = fy * (cam_points[:, 1] / np.maximum(z, 1e-6)) + cy
    valid &= (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    return u, v, valid


def write_points_txt(path: Path, header: list[str], kept_lines: list[str]) -> None:
    text = "\n".join(header + kept_lines) + "\n"
    path.write_text(text)


def write_points_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    with path.open("w") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(xyz)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property float nx\n")
        handle.write("property float ny\n")
        handle.write("property float nz\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(xyz, rgb):
            handle.write(
                f"{point[0]:.10f} {point[1]:.10f} {point[2]:.10f} 0 0 0 "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dataset", type=Path, help="Write a filtered copy instead of modifying --dataset in place.")
    parser.add_argument("--alpha-threshold", type=int, default=32)
    parser.add_argument("--min-foreground-views", type=int, default=1)
    parser.add_argument("--min-foreground-ratio", type=float, default=0.08)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if args.output_dataset is not None:
        output_dataset = args.output_dataset
        if output_dataset.exists():
            raise FileExistsError(output_dataset)
        shutil.copytree(dataset, output_dataset, copy_function=shutil.copy2)
        dataset = output_dataset

    sparse = dataset / "sparse" / "0"
    cameras = read_cameras(sparse / "cameras.txt")
    images = read_images(sparse / "images.txt")
    header, point_lines, xyz, rgb = read_points(sparse / "points3D.txt")

    visible_counts = np.zeros(len(xyz), dtype=np.int32)
    foreground_counts = np.zeros(len(xyz), dtype=np.int32)

    for pose in images:
        camera = cameras[pose.camera_id]
        alpha = np.asarray(Image.open(dataset / "images" / pose.name).convert("RGBA"))[:, :, 3]
        u, v, valid = project(xyz, pose, camera)
        if not np.any(valid):
            continue
        ui = np.clip(np.rint(u[valid]).astype(np.int32), 0, camera.width - 1)
        vi = np.clip(np.rint(v[valid]).astype(np.int32), 0, camera.height - 1)
        hits = alpha[vi, ui] >= args.alpha_threshold
        valid_indices = np.flatnonzero(valid)
        visible_counts[valid_indices] += 1
        foreground_counts[valid_indices[hits]] += 1

    ratios = foreground_counts / np.maximum(visible_counts, 1)
    keep = (foreground_counts >= args.min_foreground_views) & (ratios >= args.min_foreground_ratio)
    kept_lines = [line for line, keep_point in zip(point_lines, keep) if keep_point]

    write_points_txt(sparse / "points3D.txt", header, kept_lines)
    write_points_ply(sparse / "points3D.ply", xyz[keep], rgb[keep])

    print(
        {
            "input_points": int(len(xyz)),
            "kept_points": int(keep.sum()),
            "removed_points": int((~keep).sum()),
            "alpha_threshold": args.alpha_threshold,
            "min_foreground_views": args.min_foreground_views,
            "min_foreground_ratio": args.min_foreground_ratio,
        }
    )


if __name__ == "__main__":
    main()
