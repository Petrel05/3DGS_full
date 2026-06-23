from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .sfm import Observation, project_points


@dataclass
class BAResult:
    rotations: list[np.ndarray]
    translations: list[np.ndarray]
    points: np.ndarray
    initial_rmse: float
    final_rmse: float
    iterations: int
    accepted_steps: int
    backend: str


def _rmse(residuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(residuals)))) if residuals.size else float("inf")


def _pack(rotations: list[np.ndarray], translations: list[np.ndarray], points: np.ndarray) -> np.ndarray:
    params = []
    for r, t in zip(rotations[1:], translations[1:]):
        rvec, _ = cv2.Rodrigues(r)
        params.extend(rvec.reshape(3))
        params.extend(t.reshape(3))
    params.extend(points.reshape(-1))
    return np.array(params, dtype=np.float64)


def _unpack(params: np.ndarray, base_rotations, base_translations, n_points: int):
    rotations = [base_rotations[0]]
    translations = [base_translations[0]]
    cursor = 0
    for _ in range(1, len(base_rotations)):
        rvec = params[cursor : cursor + 3]
        t = params[cursor + 3 : cursor + 6]
        r, _ = cv2.Rodrigues(rvec)
        rotations.append(r)
        translations.append(t.copy())
        cursor += 6
    points = params[cursor : cursor + n_points * 3].reshape(n_points, 3).copy()
    return rotations, translations, points


def _residuals(params, base_rotations, base_translations, k, tracks: list[list[Observation]], n_points: int) -> np.ndarray:
    rotations, translations, points = _unpack(params, base_rotations, base_translations, n_points)
    residuals = []
    for point, track in zip(points, tracks):
        point2 = point.reshape(1, 3)
        for obs in track:
            xy = project_points(point2, k, rotations[obs.image_id], translations[obs.image_id])[0]
            residuals.extend(xy - obs.xy)
    return np.array(residuals, dtype=np.float64)


def _numeric_jacobian(fun, x: np.ndarray, fx: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    jac = np.empty((fx.size, x.size), dtype=np.float64)
    for i in range(x.size):
        step = eps * max(1.0, abs(x[i]))
        xp = x.copy()
        xp[i] += step
        jac[:, i] = (fun(xp) - fx) / step
    return jac


def _state_residuals(k, rotations, translations, points, tracks: list[list[Observation]]) -> np.ndarray:
    residuals = []
    for point, track in zip(points, tracks):
        point2 = point.reshape(1, 3)
        for obs in track:
            xy = project_points(point2, k, rotations[obs.image_id], translations[obs.image_id])[0]
            residuals.extend(xy - obs.xy)
    return np.array(residuals, dtype=np.float64)


def _refine_camera_poses(k, rotations, translations, points, tracks: list[list[Observation]]) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    opt_rotations = [r.copy() for r in rotations]
    opt_translations = [t.copy() for t in translations]
    accepted = 0
    for image_id in range(1, len(rotations)):
        object_points = []
        image_points = []
        for point, track in zip(points, tracks):
            for obs in track:
                if obs.image_id == image_id:
                    object_points.append(point)
                    image_points.append(obs.xy)
        if len(object_points) < 6:
            continue
        object_points = np.asarray(object_points, dtype=np.float64)
        image_points = np.asarray(image_points, dtype=np.float64)
        before = _rmse(_state_residuals(k, opt_rotations, opt_translations, points, tracks))
        rvec, _ = cv2.Rodrigues(opt_rotations[image_id])
        tvec = opt_translations[image_id].reshape(3, 1).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            k,
            None,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue
        try:
            rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, k, None, rvec, tvec)
        except cv2.error:
            pass
        candidate_r, _ = cv2.Rodrigues(rvec)
        old_r, old_t = opt_rotations[image_id], opt_translations[image_id]
        opt_rotations[image_id] = candidate_r
        opt_translations[image_id] = tvec.reshape(3)
        after = _rmse(_state_residuals(k, opt_rotations, opt_translations, points, tracks))
        if after < before:
            accepted += 1
        else:
            opt_rotations[image_id] = old_r
            opt_translations[image_id] = old_t
    return opt_rotations, opt_translations, accepted


def _refine_points(k, rotations, translations, points, tracks: list[list[Observation]], iterations: int) -> tuple[np.ndarray, int]:
    opt_points = points.copy()
    accepted = 0
    for point_id, track in enumerate(tracks):
        if len(track) < 2:
            continue
        x = opt_points[point_id].copy()

        def fun(z):
            residuals = []
            point2 = z.reshape(1, 3)
            for obs in track:
                xy = project_points(point2, k, rotations[obs.image_id], translations[obs.image_id])[0]
                residuals.extend(xy - obs.xy)
            return np.array(residuals, dtype=np.float64)

        residual = fun(x)
        for _ in range(iterations):
            jac = _numeric_jacobian(fun, x, residual, eps=1e-4)
            lhs = jac.T @ jac + 1e-3 * np.eye(3)
            rhs = -(jac.T @ residual)
            try:
                step = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                break
            candidate = x + step
            candidate_residual = fun(candidate)
            if _rmse(candidate_residual) < _rmse(residual):
                x = candidate
                residual = candidate_residual
                accepted += 1
            else:
                break
        opt_points[point_id] = x
    return opt_points, accepted


def bundle_adjust(
    k: np.ndarray,
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    points: np.ndarray,
    tracks: list[list[Observation]],
    max_points: int = 0,
    max_iterations: int = 6,
    backend: str = "scipy",
) -> BAResult:
    if len(points) == 0 or not tracks:
        return BAResult(rotations, translations, points, float("inf"), float("inf"), 0, 0, "none")

    order = sorted(range(len(points)), key=lambda i: (-len(tracks[i]), np.linalg.norm(points[i])))
    chosen_count = len(order) if max_points <= 0 else min(max_points, len(order))
    chosen = order[:chosen_count]
    ba_points = points[chosen].copy()
    ba_tracks = [tracks[i] for i in chosen]

    initial = _rmse(_state_residuals(k, rotations, translations, ba_points, ba_tracks))
    if backend in {"scipy", "auto"}:
        try:
            opt_rot, opt_trans, opt_points, nfev = _scipy_bundle_adjust(
                k,
                rotations,
                translations,
                ba_points,
                ba_tracks,
                max_iterations=max_iterations,
            )
            all_points = points.copy()
            all_points[chosen] = opt_points
            final = _rmse(_state_residuals(k, opt_rot, opt_trans, opt_points, ba_tracks))
            return BAResult(
                rotations=opt_rot,
                translations=opt_trans,
                points=all_points,
                initial_rmse=initial,
                final_rmse=final,
                iterations=max_iterations,
                accepted_steps=int(nfev),
                backend="scipy.optimize.least_squares",
            )
        except Exception:
            if backend == "scipy":
                raise

    opt_rot, opt_trans, pose_steps = _refine_camera_poses(k, rotations, translations, ba_points, ba_tracks)
    opt_points, point_steps = _refine_points(k, opt_rot, opt_trans, ba_points, ba_tracks, iterations=max_iterations)
    all_points = points.copy()
    all_points[chosen] = opt_points
    final = _rmse(_state_residuals(k, opt_rot, opt_trans, opt_points, ba_tracks))
    return BAResult(
        rotations=opt_rot,
        translations=opt_trans,
        points=all_points,
        initial_rmse=initial,
        final_rmse=final,
        iterations=max_iterations,
        accepted_steps=pose_steps + point_steps,
        backend="opencv-pnp-plus-numeric-gn",
    )


def _scipy_bundle_adjust(
    k: np.ndarray,
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    points: np.ndarray,
    tracks: list[list[Observation]],
    max_iterations: int,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, int]:
    from scipy.optimize import least_squares

    params0 = _pack(rotations, translations, points)

    def fun(params: np.ndarray) -> np.ndarray:
        return _residuals(params, rotations, translations, k, tracks, len(points))

    result = least_squares(
        fun,
        params0,
        method="trf",
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=None if max_iterations <= 0 else max(10, max_iterations * 20),
        x_scale="jac",
        verbose=0,
    )
    opt_rot, opt_trans, opt_points = _unpack(result.x, rotations, translations, len(points))
    return opt_rot, opt_trans, opt_points, int(result.nfev)
