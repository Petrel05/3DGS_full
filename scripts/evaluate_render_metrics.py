from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def natural_key(path: Path) -> list[object]:
    import re

    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def read_alpha(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 3 or image.shape[2] < 4:
        return None
    return image[:, :, 3].astype(np.float32) / 255.0


def psnr(render: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = render - gt
    if mask is not None:
        valid = mask > 0.5
        if not np.any(valid):
            return float("nan")
        diff = diff[valid]
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def ssim_gray(render: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    gray_r = cv2.cvtColor((np.clip(render, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_g = cv2.cvtColor((np.clip(gt, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    c1 = 0.01**2
    c2 = 0.03**2
    kernel = (11, 11)
    mu_r = cv2.GaussianBlur(gray_r, kernel, 1.5)
    mu_g = cv2.GaussianBlur(gray_g, kernel, 1.5)
    sigma_r = cv2.GaussianBlur(gray_r * gray_r, kernel, 1.5) - mu_r * mu_r
    sigma_g = cv2.GaussianBlur(gray_g * gray_g, kernel, 1.5) - mu_g * mu_g
    sigma_rg = cv2.GaussianBlur(gray_r * gray_g, kernel, 1.5) - mu_r * mu_g
    value = ((2 * mu_r * mu_g + c1) * (2 * sigma_rg + c2)) / (
        (mu_r * mu_r + mu_g * mu_g + c1) * (sigma_r + sigma_g + c2) + 1e-12
    )
    if mask is not None:
        valid = mask > 0.5
        if np.any(valid):
            value = value[valid]
    return float(np.mean(value))


def evaluate(render_dir: Path, gt_dir: Path) -> dict:
    render_paths = sorted([p for p in render_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}], key=natural_key)
    rows = []
    for render_path in render_paths:
        gt_path = gt_dir / render_path.name
        if not gt_path.exists():
            # Graphdeco often names render files 00000.png while GT uses the same.
            continue
        render = read_rgb(render_path)
        gt = read_rgb(gt_path)
        alpha = read_alpha(gt_path)
        if render.shape != gt.shape:
            gt = cv2.resize(gt, (render.shape[1], render.shape[0]), interpolation=cv2.INTER_AREA)
            if alpha is not None:
                alpha = cv2.resize(alpha, (render.shape[1], render.shape[0]), interpolation=cv2.INTER_NEAREST)
        foreground = alpha if alpha is not None else np.ones(render.shape[:2], dtype=np.float32)
        background = 1.0 - foreground
        fg_valid = foreground > 0.5
        bg_valid = background > 0.5
        abs_err = np.abs(render - gt)
        rows.append(
            {
                "image": render_path.name,
                "psnr": psnr(render, gt),
                "fg_psnr": psnr(render, gt, foreground),
                "ssim": ssim_gray(render, gt),
                "fg_ssim": ssim_gray(render, gt, foreground),
                "fg_mae": float(np.mean(abs_err[fg_valid])) if np.any(fg_valid) else None,
                "bg_render_mean": float(np.mean(render[bg_valid])) if np.any(bg_valid) else None,
                "bg_render_p95": float(np.percentile(render[bg_valid], 95)) if np.any(bg_valid) else None,
                "fg_pixels": int(fg_valid.sum()),
                "bg_pixels": int(bg_valid.sum()),
            }
        )
    if not rows:
        raise RuntimeError(f"No matching render/GT files found: {render_dir} vs {gt_dir}")

    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key not in {"fg_pixels", "bg_pixels"}]
    summary = {"count": len(rows), "per_image": rows}
    for key in numeric_keys:
        values = [row[key] for row in rows if row[key] is not None and np.isfinite(row[key])]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else None
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args.render_dir, args.gt_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_image"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
