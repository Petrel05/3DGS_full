from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .io_utils import Frame, ensure_dir, load_image_sequence, natural_key
from .sfm import Reconstruction, reconstruct_with_known_cameras


@dataclass
class VGGTOutput:
    reconstruction: Reconstruction
    raw_points: np.ndarray
    raw_colors: np.ndarray
    confidences: np.ndarray
    backend: str
    image_paths: list[Path]
    frames: list[Frame]


def _write_vggt_inputs(frames: list[Frame], output_dir: Path) -> list[Path]:
    input_dir = ensure_dir(output_dir / "vggt_inputs")
    paths = []
    for i, frame in enumerate(frames):
        path = input_dir / f"rgb_{i:04d}.png"
        cv2.imwrite(str(path), frame.image)
        paths.append(path)
    return paths


def _sample_points(
    points: np.ndarray,
    colors: np.ndarray,
    confidences: np.ndarray,
    max_points: int,
    percentile: float,
    foreground: np.ndarray | None = None,
    frame_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidences)
    if foreground is not None:
        finite &= foreground.astype(bool)
    if frame_ids is not None:
        frame_ids = frame_ids[finite]
    points = points[finite]
    colors = colors[finite]
    confidences = confidences[finite]
    if len(points) == 0:
        return points, colors, confidences
    if percentile > 0:
        threshold = np.percentile(confidences, percentile)
        keep = confidences >= threshold
        points = points[keep]
        colors = colors[keep]
        confidences = confidences[keep]
        if frame_ids is not None:
            frame_ids = frame_ids[keep]
    if max_points > 0 and len(points) > max_points:
        if frame_ids is not None:
            chosen = _view_balanced_spatial_sample(points, confidences, frame_ids, max_points)
        else:
            chosen = _spatial_confidence_sample(points, confidences, max_points)
        points = points[chosen]
        colors = colors[chosen]
        confidences = confidences[chosen]
    return points.astype(np.float64), colors.astype(np.float64), confidences.astype(np.float64)


def _spatial_confidence_sample(points: np.ndarray, confidences: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return np.arange(len(points))

    lower = np.percentile(points, 1, axis=0)
    upper = np.percentile(points, 99, axis=0)
    span = np.maximum(upper - lower, 1e-6)
    grid_size = max(8, int(np.ceil((max_points * 4) ** (1.0 / 3.0))))
    coords = np.floor((np.clip(points, lower, upper) - lower) / span * (grid_size - 1)).astype(np.int64)
    keys = coords[:, 0] + grid_size * coords[:, 1] + (grid_size * grid_size) * coords[:, 2]

    # Keep the highest-confidence point in each occupied voxel, then thin voxels
    # uniformly in space if the viewer budget is still exceeded.
    order = np.lexsort((-confidences, keys))
    _, first = np.unique(keys[order], return_index=True)
    chosen = order[first]
    chosen = chosen[np.argsort(keys[chosen])]
    if len(chosen) > max_points:
        chosen = chosen[np.linspace(0, len(chosen) - 1, max_points, dtype=np.int64)]
    elif len(chosen) < max_points:
        chosen_mask = np.zeros(len(points), dtype=bool)
        chosen_mask[chosen] = True
        remaining = np.argsort(confidences[~chosen_mask])[::-1]
        remaining_indices = np.flatnonzero(~chosen_mask)[remaining]
        chosen = np.concatenate([chosen, remaining_indices[: max_points - len(chosen)]])
    return chosen


def _view_balanced_spatial_sample(
    points: np.ndarray,
    confidences: np.ndarray,
    frame_ids: np.ndarray,
    max_points: int,
) -> np.ndarray:
    unique_frames = np.unique(frame_ids)
    if len(unique_frames) <= 1:
        return _spatial_confidence_sample(points, confidences, max_points)

    quota = int(np.ceil(max_points / len(unique_frames)))
    chosen_parts = []
    for frame_id in unique_frames:
        frame_indices = np.flatnonzero(frame_ids == frame_id)
        if len(frame_indices) == 0:
            continue
        frame_quota = min(quota, len(frame_indices))
        local = _spatial_confidence_sample(points[frame_indices], confidences[frame_indices], frame_quota)
        chosen_parts.append(frame_indices[local])

    if not chosen_parts:
        return _spatial_confidence_sample(points, confidences, max_points)

    chosen = np.concatenate(chosen_parts)
    if len(chosen) > max_points:
        local = _spatial_confidence_sample(points[chosen], confidences[chosen], max_points)
        return chosen[local]

    if len(chosen) < max_points:
        chosen_mask = np.zeros(len(points), dtype=bool)
        chosen_mask[chosen] = True
        remaining_indices = np.flatnonzero(~chosen_mask)
        need = min(max_points - len(chosen), len(remaining_indices))
        if need > 0:
            local = _spatial_confidence_sample(points[remaining_indices], confidences[remaining_indices], need)
            chosen = np.concatenate([chosen, remaining_indices[local]])
    return chosen


def _preprocess_mask(mask: np.ndarray | None, image_shape: tuple[int, int, int], mode: str) -> np.ndarray:
    target_size = 518
    height, width = image_shape[:2]
    if mask is None:
        mask = np.ones((height, width), dtype=np.uint8) * 255
    pil = Image.fromarray(mask.astype(np.uint8), mode="L")
    if mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
    elif mode == "crop":
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    pil = pil.resize((new_width, new_height), Image.Resampling.NEAREST)
    arr = np.asarray(pil, dtype=np.uint8)
    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        arr = arr[start_y : start_y + target_size, :]
    if mode == "pad":
        h_padding = target_size - arr.shape[0]
        w_padding = target_size - arr.shape[1]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=0)
    return arr > 10


def _preprocess_masks(frames: list[Frame], mode: str) -> np.ndarray:
    masks = [_preprocess_mask(frame.mask, frame.image.shape, mode) for frame in frames]
    max_h = max(mask.shape[0] for mask in masks)
    max_w = max(mask.shape[1] for mask in masks)
    padded = []
    for mask in masks:
        h_padding = max_h - mask.shape[0]
        w_padding = max_w - mask.shape[1]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=False)
        padded.append(mask)
    return np.stack(padded, axis=0)


def _processed_frames(input_images: np.ndarray, mask_stack: np.ndarray, names: list[str]) -> list[Frame]:
    images_rgb = np.transpose(input_images, (0, 2, 3, 1))
    frames = []
    for image_rgb, mask, name in zip(images_rgb, mask_stack, names):
        image_bgr = np.clip(image_rgb[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
        mask_u8 = mask.astype(np.uint8) * 255
        frames.append(Frame(name=name, image=image_bgr, mask=mask_u8, scale=1.0))
    return frames


def run_vggt(
    source: str | Path,
    output_dir: str | Path,
    model_path: str | Path = ".models/VGGT-1B",
    max_images: int = 0,
    max_size: int = 0,
    max_features: int = 0,
    max_points: int = 0,
    confidence_percentile: float = 0.0,
    mode: str = "pad",
    allow_cpu: bool = False,
    skip_sparse: bool = False,
    sparse_match_window: int = 1,
    sparse_loop_closure: bool = False,
) -> VGGTOutput:
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    output_dir = ensure_dir(output_dir)
    frames = load_image_sequence(source, max_images=max_images, max_size=max_size, use_masks=True)
    image_paths = _write_vggt_inputs(frames, output_dir)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu" and not allow_cpu:
        raise RuntimeError("VGGT-1B is installed, but no CUDA/MPS accelerator is available. Use --allow-cpu-vggt to force slow CPU inference.")
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

    model_ref = Path(model_path)
    model_name = str(model_ref) if model_ref.exists() else str(model_path)
    model = VGGT.from_pretrained(model_name).to(device)
    model.eval()
    images = load_and_preprocess_images([str(p) for p in image_paths], mode=mode).to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype, enabled=device != "cpu"):
            predictions = model(images)

    h, w = int(images.shape[-2]), int(images.shape[-1])
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        predictions["pose_enc"], image_size_hw=(h, w), build_intrinsics=True
    )
    extrinsics_np = extrinsics[0].detach().cpu().numpy()
    intrinsics_np = intrinsics[0].detach().cpu().numpy()
    world_points = predictions["world_points"][0].detach().cpu().numpy()
    world_conf = predictions["world_points_conf"][0].detach().cpu().numpy()
    input_images = predictions["images"][0].detach().cpu().numpy()

    flat_points = world_points.reshape(-1, 3)
    flat_conf = world_conf.reshape(-1)
    flat_colors = (np.transpose(input_images, (0, 2, 3, 1)).reshape(-1, 3) * 255.0)[:, ::-1]
    mask_stack = _preprocess_masks(frames, mode=mode)
    foreground = mask_stack.reshape(-1)
    frame_ids = np.repeat(np.arange(world_points.shape[0]), world_points.shape[1] * world_points.shape[2])
    points, colors, confidences = _sample_points(
        flat_points,
        flat_colors,
        flat_conf,
        max_points=max_points,
        percentile=confidence_percentile,
        foreground=foreground,
        frame_ids=frame_ids,
    )

    rotations = [ext[:3, :3].astype(np.float64) for ext in extrinsics_np]
    translations = [ext[:3, 3].astype(np.float64) for ext in extrinsics_np]
    ba_frames = _processed_frames(input_images, mask_stack, [path.name for path in image_paths])
    if skip_sparse:
        sparse_reconstruction = Reconstruction(
            camera_matrix=intrinsics_np[0].astype(np.float64),
            rotations=rotations,
            translations=translations,
            points=np.zeros((0, 3), dtype=np.float64),
            colors=np.zeros((0, 3), dtype=np.float64),
            tracks=[],
            reprojection_rmse=float("nan"),
            pair_inliers=[],
        )
    else:
        sparse_reconstruction = reconstruct_with_known_cameras(
            ba_frames,
            intrinsics_np[0].astype(np.float64),
            rotations,
            translations,
            max_features=max_features,
            max_reprojection=12.0,
            match_window=sparse_match_window,
            loop_closure=sparse_loop_closure,
        )
    return VGGTOutput(
        reconstruction=sparse_reconstruction,
        raw_points=points,
        raw_colors=colors,
        confidences=confidences,
        backend=f"VGGT on {device}",
        image_paths=sorted(image_paths, key=natural_key),
        frames=ba_frames,
    )
