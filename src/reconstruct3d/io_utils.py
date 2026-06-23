from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Frame:
    name: str
    image: np.ndarray
    mask: np.ndarray | None
    scale: float


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def natural_key(path: Path) -> list[object]:
    import re

    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _resize_keep_aspect(image: np.ndarray, max_size: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if max_size <= 0:
        return image, 1.0
    scale = min(1.0, float(max_size) / max(h, w))
    if scale == 1.0:
        return image, 1.0
    resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def _green_screen_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
    mask = cv2.bitwise_not(green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _mask_for_image(image_path: Path) -> Path | None:
    if image_path.name.startswith("rgb_"):
        candidate = image_path.with_name(image_path.name.replace("rgb_", "msk_", 1))
        if candidate.exists():
            return candidate
    stem_candidate = image_path.with_name(f"msk_{image_path.stem.split('_')[-1]}.png")
    return stem_candidate if stem_candidate.exists() else None


def iter_rgb_images(directory: str | Path) -> Iterable[Path]:
    directory = Path(directory)
    files = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    rgb = [p for p in files if not p.name.startswith("msk_")]
    return sorted(rgb, key=natural_key)


def load_image_sequence(
    source: str | Path,
    max_images: int = 16,
    max_size: int = 900,
    use_masks: bool = True,
) -> list[Frame]:
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        frame_dir = extract_video_frames(source, source.parent / f"{source.stem}_frames", max_images=max_images)
        image_paths = list(iter_rgb_images(frame_dir))
    else:
        image_paths = list(iter_rgb_images(source))
    if max_images > 0:
        image_paths = image_paths[:max_images]
    if len(image_paths) < 2:
        raise ValueError(f"Need at least two input images, got {len(image_paths)} from {source}")

    frames: list[Frame] = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        image, scale = _resize_keep_aspect(image, max_size)

        mask = None
        mask_path = _mask_for_image(path) if use_masks else None
        if mask_path is not None:
            raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if raw_mask is not None:
                raw_mask, _ = _resize_keep_aspect(raw_mask, max_size)
                mask = (raw_mask > 10).astype(np.uint8) * 255
        if mask is None and use_masks:
            mask = _green_screen_mask(image)
        frames.append(Frame(name=path.name, image=image, mask=mask, scale=scale))
    return frames


def extract_video_frames(video_path: str | Path, output_dir: str | Path, max_images: int = 16) -> Path:
    video_path = Path(video_path)
    output_dir = ensure_dir(output_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    count = max(2, max_images)
    indices = np.linspace(0, max(total - 1, 1), count, dtype=int)
    for out_idx, frame_idx in enumerate(indices):
        target = output_dir / f"rgb_{out_idx:04d}.png"
        if target.exists():
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(target), frame)
    cap.release()
    return output_dir


def camera_matrix(width: int, height: int, focal_scale: float = 1.2) -> np.ndarray:
    focal = focal_scale * max(width, height)
    return np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    path = Path(path)
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[2])} {int(c[1])} {int(c[0])}\n")
