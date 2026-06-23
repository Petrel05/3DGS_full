from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class GaussianBuildResult:
    data: dict
    backend: str
    initial_loss: float | None = None
    final_loss: float | None = None
    iterations: int = 0


def pairwise_nearest(points: np.ndarray, max_neighbors: int = 8) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.zeros((0, 0), dtype=int), np.zeros((0, 0), dtype=np.float64)
    if len(points) > 5000:
        try:
            from scipy.spatial import cKDTree

            k = min(max_neighbors + 1, len(points))
            dist, idx = cKDTree(points).query(points, k=k, workers=-1)
            if k == 1:
                return np.zeros((len(points), 0), dtype=int), np.zeros((len(points), 0), dtype=np.float64)
            return idx[:, 1:].astype(int), dist[:, 1:].astype(np.float64)
        except Exception:
            sample_count = min(5000, len(points))
            sample = np.linspace(0, len(points) - 1, sample_count, dtype=int)
            sample_points = points[sample]
            diff = points[:, None, :] - sample_points[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            same = sample[None, :] == np.arange(len(points))[:, None]
            dist[same] = np.inf
            k = min(max_neighbors, sample_count)
            sample_idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
            row = np.arange(len(points))[:, None]
            vals = dist[row, sample_idx]
            order = np.argsort(vals, axis=1)
            return sample[sample_idx[row, order]], vals[row, order]
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    k = min(max_neighbors, max(1, len(points) - 1))
    idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    row = np.arange(len(points))[:, None]
    vals = dist[row, idx]
    order = np.argsort(vals, axis=1)
    return idx[row, order], vals[row, order]


def normalize_points(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    center = np.median(points, axis=0)
    centered = points - center
    scale = np.percentile(np.linalg.norm(centered, axis=1), 90)
    if scale <= 1e-8:
        scale = 1.0
    return centered / scale


def estimate_radius(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.015
    sample = points[: min(300, len(points))]
    diff = sample[:, None, :] - sample[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist[dist == 0] = np.inf
    nearest = np.min(dist, axis=1)
    radius = float(np.clip(np.median(nearest) * 1.8, 0.006, 0.08))
    return radius


def optimize_gaussian_parameters(
    points: np.ndarray,
    colors_bgr: np.ndarray,
    iterations: int = 4,
    max_neighbors: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, colors_bgr, np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    points = normalize_points(points)
    colors = np.clip(colors_bgr.astype(np.float64), 0, 255)
    neighbor_idx, neighbor_dist = pairwise_nearest(points, max_neighbors=max_neighbors)
    median_dist = np.median(neighbor_dist, axis=1) if neighbor_dist.size else np.ones((len(points),))
    valid = median_dist < np.percentile(median_dist, 95)
    points = points[valid]
    colors = colors[valid]
    if len(points) == 0:
        return points, colors, np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    for _ in range(iterations):
        neighbor_idx, neighbor_dist = pairwise_nearest(points, max_neighbors=max_neighbors)
        weights = 1.0 / np.maximum(neighbor_dist, 1e-4)
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        neighbor_colors = colors[neighbor_idx]
        smoothed = (neighbor_colors * weights[..., None]).sum(axis=1)
        colors = 0.82 * colors + 0.18 * smoothed

    neighbor_idx, neighbor_dist = pairwise_nearest(points, max_neighbors=max_neighbors)
    local_scale = np.median(neighbor_dist, axis=1) if neighbor_dist.size else np.full((len(points),), 0.02)
    radii = np.clip(local_scale * 1.6, 0.006, 0.075)
    density = 1.0 / np.maximum(local_scale, 1e-4)
    density = (density - density.min()) / max(float(density.max() - density.min()), 1e-8)
    opacities = np.clip(0.35 + 0.55 * density, 0.35, 0.90)
    return points, colors, radii, opacities


def build_gaussians(points: np.ndarray, colors_bgr: np.ndarray, max_points: int = 0) -> dict:
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
        colors_bgr = colors_bgr[indices]
    norm_points, opt_colors, radii, opacities = optimize_gaussian_parameters(points, colors_bgr)
    colors_rgb = opt_colors[:, ::-1] / 255.0 if len(opt_colors) else np.zeros((0, 3))
    return {
        "points": norm_points.round(6).tolist(),
        "colors": colors_rgb.round(4).tolist(),
        "radii": radii.round(5).tolist(),
        "opacities": opacities.round(3).tolist(),
        "note": "Gaussian points optimized by density-based outlier pruning, kNN radius fitting, opacity estimation, and local color smoothing.",
    }


def build_gaussians_optimized(
    points: np.ndarray,
    colors_bgr: np.ndarray,
    frames: list[Any] | None = None,
    camera_matrix: np.ndarray | None = None,
    rotations: list[np.ndarray] | None = None,
    translations: list[np.ndarray] | None = None,
    max_points: int = 0,
    backend: str = "auto",
    iterations: int = 80,
    image_size: int = 0,
    max_views: int = 0,
) -> GaussianBuildResult:
    if backend == "knn" or frames is None or camera_matrix is None or rotations is None or translations is None:
        return GaussianBuildResult(build_gaussians(points, colors_bgr, max_points=max_points), "knn-density")
    if backend in {"auto", "gsplat"}:
        try:
            data, initial_loss, final_loss = optimize_gaussians_gsplat(
                points,
                colors_bgr,
                frames,
                camera_matrix,
                rotations,
                translations,
                max_points=max_points,
                iterations=iterations,
                image_size=image_size,
                max_views=max_views,
            )
            return GaussianBuildResult(
                data=data,
                backend="gsplat-rasterization",
                initial_loss=initial_loss,
                final_loss=final_loss,
                iterations=iterations,
            )
        except Exception:
            if backend == "gsplat":
                raise
    try:
        data, initial_loss, final_loss = optimize_gaussians_torch(
            points,
            colors_bgr,
            frames,
            camera_matrix,
            rotations,
            translations,
            max_points=max_points,
            iterations=iterations,
            image_size=image_size,
            max_views=max_views,
        )
        return GaussianBuildResult(
            data=data,
            backend="torch-autograd-photometric",
            initial_loss=initial_loss,
            final_loss=final_loss,
            iterations=iterations,
        )
    except Exception:
        if backend == "torch":
            raise
        return GaussianBuildResult(build_gaussians(points, colors_bgr, max_points=max_points), "knn-density-fallback")


def _camera_batches(camera_matrix: np.ndarray, rotations, translations, chosen: list[int], scales: list[float], device: str):
    import torch

    viewmats = []
    intrinsics = []
    for image_id, scale in zip(chosen, scales):
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = rotations[image_id].astype(np.float32)
        view[:3, 3] = translations[image_id].astype(np.float32)
        k = camera_matrix.astype(np.float32).copy()
        k[:2, :] *= scale
        viewmats.append(view)
        intrinsics.append(k)
    return (
        torch.tensor(np.stack(viewmats), dtype=torch.float32, device=device),
        torch.tensor(np.stack(intrinsics), dtype=torch.float32, device=device),
    )


def _quat_identity(count: int, device: str):
    import torch

    quats = torch.zeros((count, 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0
    return quats


def optimize_gaussians_gsplat(
    points: np.ndarray,
    colors_bgr: np.ndarray,
    frames: list[Any],
    camera_matrix: np.ndarray,
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    max_points: int = 0,
    iterations: int = 80,
    image_size: int = 0,
    max_views: int = 0,
) -> tuple[dict, float, float]:
    import torch
    from gsplat.rendering import rasterization

    if not torch.cuda.is_available():
        raise RuntimeError("gsplat rasterization requires CUDA in this project configuration.")
    if len(points) == 0:
        data = build_gaussians(points, colors_bgr, max_points=max_points)
        return data, 0.0, 0.0
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
        colors_bgr = colors_bgr[indices]

    device = "cuda"
    chosen, targets, masks, scales = _prepare_training_views(frames, max_views=max_views, image_size=image_size)
    render_extent = max(max(item.shape[:2]) for item in targets) if targets else max(float(image_size), 1.0)
    targets = [item.to(device) for item in targets]
    masks = [item.to(device) for item in masks]
    target_batch = torch.stack(targets, dim=0)
    mask_batch = torch.stack(masks, dim=0)
    height, width = target_batch.shape[1:3]
    viewmats, intrinsics = _camera_batches(camera_matrix, rotations, translations, chosen, scales, device)

    xyz0 = torch.tensor(points, dtype=torch.float32, device=device)
    rgb0 = torch.tensor(np.clip(colors_bgr[:, ::-1] / 255.0, 1e-4, 1.0 - 1e-4), dtype=torch.float32, device=device)
    color_logits0 = torch.logit(rgb0)
    scale0 = estimate_radius(points)

    means = torch.nn.Parameter(xyz0.clone())
    scales_param = torch.nn.Parameter(torch.full((len(points), 3), float(scale0), dtype=torch.float32, device=device))
    quats = torch.nn.Parameter(_quat_identity(len(points), device))
    opacity_logits = torch.nn.Parameter(torch.full((len(points),), 0.0, dtype=torch.float32, device=device))
    color_logits = torch.nn.Parameter(color_logits0.clone())
    optimizer = torch.optim.Adam(
        [
            {"params": [means], "lr": 5e-4},
            {"params": [scales_param], "lr": 2e-3},
            {"params": [quats], "lr": 5e-4},
            {"params": [opacity_logits], "lr": 3e-2},
            {"params": [color_logits], "lr": 4e-2},
        ]
    )

    def render() -> "torch.Tensor":
        colors = torch.sigmoid(color_logits)
        opacities = torch.sigmoid(opacity_logits)
        scales_pos = torch.nn.functional.softplus(scales_param).clamp(1e-4, 0.25)
        quats_norm = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        rendered, _, _ = rasterization(
            means,
            quats_norm,
            scales_pos,
            opacities,
            colors,
            viewmats,
            intrinsics,
            width=width,
            height=height,
            render_mode="RGB",
            packed=True,
        )
        return rendered[..., :3].clamp(0.0, 1.0)

    def compute_loss() -> "torch.Tensor":
        pred = render()
        photometric = (((pred - target_batch) * mask_batch) ** 2).sum() / mask_batch.sum().clamp_min(1.0)
        scale_reg = 0.001 * torch.mean(torch.nn.functional.softplus(scales_param) ** 2)
        opacity_reg = 0.001 * torch.mean(torch.sigmoid(opacity_logits))
        return photometric + scale_reg + opacity_reg

    with torch.no_grad():
        initial_loss = float(compute_loss().detach().cpu())
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float(compute_loss().detach().cpu())
        opt_points = means.detach().cpu().numpy()
        norm_points = normalize_points(opt_points)
        colors_rgb = torch.sigmoid(color_logits).detach().cpu().numpy()
        radii = np.clip(torch.nn.functional.softplus(scales_param).mean(dim=1).detach().cpu().numpy(), 0.006, 0.08)
        opacities = np.clip(torch.sigmoid(opacity_logits).detach().cpu().numpy(), 0.05, 0.98)
    data = {
        "points": norm_points.round(6).tolist(),
        "colors": colors_rgb.round(4).tolist(),
        "radii": radii.round(5).tolist(),
        "opacities": opacities.round(3).tolist(),
        "note": "Gaussian parameters optimized with the open-source gsplat CUDA rasterizer.",
    }
    return data, initial_loss, final_loss


def _prepare_training_views(frames: list[Any], max_views: int, image_size: int):
    import cv2
    import torch

    if max_views <= 0 or len(frames) <= max_views:
        chosen = list(range(len(frames)))
    else:
        chosen = np.linspace(0, len(frames) - 1, max_views, dtype=int).tolist()
    targets = []
    masks = []
    scales = []
    for image_id in chosen:
        frame = frames[image_id]
        h, w = frame.image.shape[:2]
        if image_size <= 0:
            scale = 1.0
            out_w, out_h = w, h
            image = frame.image
        else:
            scale = float(image_size) / float(max(h, w))
            out_w = max(8, int(round(w * scale)))
            out_h = max(8, int(round(h * scale)))
            image = cv2.resize(frame.image, (out_w, out_h), interpolation=cv2.INTER_AREA)
        target = torch.tensor(image[:, :, ::-1].copy(), dtype=torch.float32) / 255.0
        if frame.mask is None:
            mask = torch.ones((out_h, out_w, 1), dtype=torch.float32)
        else:
            raw_mask = cv2.resize(frame.mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            mask = torch.tensor((raw_mask > 10).astype(np.float32)[:, :, None])
        targets.append(target)
        masks.append(mask)
        scales.append(scale)
    return chosen, targets, masks, scales


def _initial_pixel_radius(points: np.ndarray, k: np.ndarray, rotations, translations, image_scale: float) -> float:
    if len(points) < 2:
        return 2.0
    r = rotations[0]
    t = translations[0]
    cam = (r @ points.T).T + t.reshape(1, 3)
    valid = cam[:, 2] > 1e-5
    if valid.sum() < 2:
        return 2.0
    pix = (k @ cam[valid].T).T
    xy = pix[:, :2] / np.maximum(pix[:, 2:3], 1e-8)
    sample = xy[: min(300, len(xy))] * image_scale
    diff = sample[:, None, :] - sample[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist[dist == 0] = np.inf
    nearest = np.min(dist, axis=1)
    return float(np.clip(np.median(nearest) * 0.7, 1.2, 5.0))


def optimize_gaussians_torch(
    points: np.ndarray,
    colors_bgr: np.ndarray,
    frames: list[Any],
    camera_matrix: np.ndarray,
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    max_points: int = 0,
    iterations: int = 80,
    image_size: int = 0,
    max_views: int = 0,
) -> tuple[dict, float, float]:
    import torch

    if len(points) == 0:
        data = build_gaussians(points, colors_bgr, max_points=max_points)
        return data, 0.0, 0.0
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
        colors_bgr = colors_bgr[indices]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    chosen, targets, masks, scales = _prepare_training_views(frames, max_views=max_views, image_size=image_size)
    render_extent = max(max(item.shape[:2]) for item in targets) if targets else max(float(image_size), 1.0)
    targets = [item.to(device) for item in targets]
    masks = [item.to(device) for item in masks]

    xyz0 = torch.tensor(points, dtype=torch.float32, device=device)
    rgb0 = torch.tensor(np.clip(colors_bgr[:, ::-1] / 255.0, 1e-4, 1.0 - 1e-4), dtype=torch.float32, device=device)
    color_logits0 = torch.logit(rgb0)
    first_scale = scales[0] if scales else 1.0
    radius0 = _initial_pixel_radius(points, camera_matrix, rotations, translations, first_scale)

    xyz = torch.nn.Parameter(xyz0.clone())
    color_logits = torch.nn.Parameter(color_logits0.clone())
    log_radius = torch.nn.Parameter(torch.full((len(points),), np.log(radius0), dtype=torch.float32, device=device))
    opacity_logits = torch.nn.Parameter(torch.full((len(points),), 0.4, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(
        [
            {"params": [xyz], "lr": 3e-4},
            {"params": [color_logits], "lr": 4e-2},
            {"params": [log_radius], "lr": 2e-2},
            {"params": [opacity_logits], "lr": 2e-2},
        ]
    )

    k_base = torch.tensor(camera_matrix, dtype=torch.float32, device=device)
    rot_t = [torch.tensor(rotations[i], dtype=torch.float32, device=device) for i in chosen]
    trans_t = [torch.tensor(translations[i], dtype=torch.float32, device=device) for i in chosen]

    def render_view(view_idx: int) -> "torch.Tensor":
        target = targets[view_idx]
        h, w = target.shape[:2]
        scale = scales[view_idx]
        k = k_base.clone()
        k[:2, :] *= scale
        cam = xyz @ rot_t[view_idx].T + trans_t[view_idx].reshape(1, 3)
        z = cam[:, 2].clamp_min(1e-5)
        pix = cam @ k.T
        xy = pix[:, :2] / z[:, None]
        valid = ((cam[:, 2] > 1e-5) & (xy[:, 0] >= -16) & (xy[:, 0] <= w + 16) & (xy[:, 1] >= -16) & (xy[:, 1] <= h + 16)).float()

        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing="ij",
        )
        dx = xx[None, :, :] - xy[:, 0, None, None]
        dy = yy[None, :, :] - xy[:, 1, None, None]
        sigma = torch.exp(log_radius).clamp(0.7, 8.0)[:, None, None]
        alpha = torch.sigmoid(opacity_logits).clamp(0.03, 0.98)[:, None, None]
        weights = torch.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)) * alpha * valid[:, None, None]
        colors = torch.sigmoid(color_logits)[:, :, None, None]
        weighted = (weights[:, None, :, :] * colors).sum(dim=0)
        accum = weights.sum(dim=0).clamp_min(1e-4)
        image = (weighted / accum[None, :, :]).permute(1, 2, 0)
        return image.clamp(0.0, 1.0)

    def compute_loss() -> "torch.Tensor":
        loss = torch.zeros((), dtype=torch.float32, device=device)
        for view_idx, target in enumerate(targets):
            pred = render_view(view_idx)
            mask = masks[view_idx]
            loss = loss + (((pred - target) * mask) ** 2).sum() / mask.sum().clamp_min(1.0)
        radius_reg = 0.002 * torch.mean((torch.exp(log_radius).clamp(0.7, 8.0) - radius0) ** 2)
        opacity_reg = 0.001 * torch.mean(torch.sigmoid(opacity_logits))
        return loss / max(1, len(targets)) + radius_reg + opacity_reg

    with torch.no_grad():
        initial_loss = float(compute_loss().detach().cpu())
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float(compute_loss().detach().cpu())
        opt_points = xyz.detach().cpu().numpy()
        opt_colors_bgr = (torch.sigmoid(color_logits).detach().cpu().numpy()[:, ::-1] * 255.0)
        norm_points = normalize_points(opt_points)
        radii_px = torch.exp(log_radius).detach().cpu().numpy()
        radii = np.clip(radii_px / max(float(render_extent), 1.0), 0.006, 0.08)
        opacities = np.clip(torch.sigmoid(opacity_logits).detach().cpu().numpy(), 0.05, 0.98)
        colors_rgb = np.clip(opt_colors_bgr[:, ::-1] / 255.0, 0.0, 1.0)
    data = {
        "points": norm_points.round(6).tolist(),
        "colors": colors_rgb.round(4).tolist(),
        "radii": radii.round(5).tolist(),
        "opacities": opacities.round(3).tolist(),
        "note": "Gaussian points optimized with PyTorch autograd photometric fitting over calibrated multi-view images.",
    }
    return data, initial_loss, final_loss


def write_gaussians(path: str | Path, gaussians: dict) -> None:
    Path(path).write_text(json.dumps(gaussians, ensure_ascii=False, indent=2), encoding="utf-8")
