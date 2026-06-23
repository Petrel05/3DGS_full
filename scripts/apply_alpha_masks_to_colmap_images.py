from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reconstruct3d.io_utils import load_image_sequence


def refine_alpha(mask: np.ndarray, erode: int, dilate: int, feather: int) -> np.ndarray:
    alpha = (((mask > 10).astype(np.uint8) * 255).astype(np.uint8)) if mask is not None else mask
    if alpha is None:
        raise ValueError("mask must not be None")
    if erode > 0:
        size = erode * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        alpha = cv2.erode(alpha, kernel, iterations=1)
    if dilate > 0:
        size = dilate * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        alpha = cv2.dilate(alpha, kernel, iterations=1)
    if feather > 0:
        size = feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (size, size), sigmaX=max(1.0, feather / 2.0))
    return alpha.astype(np.uint8)


def despill_green_bgr(image: np.ndarray, strength: float, bias: float) -> np.ndarray:
    bgr = image.astype(np.float32)
    blue = bgr[:, :, 0]
    green = bgr[:, :, 1]
    red = bgr[:, :, 2]
    limit = np.maximum(red, blue) * strength + bias
    bgr[:, :, 1] = np.minimum(green, limit)
    return np.clip(bgr, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace COLMAP training images with RGBA images using foreground masks.")
    parser.add_argument("--source", required=True, help="Original image directory or video.")
    parser.add_argument("--dataset", required=True, type=Path, help="COLMAP dataset root containing images/.")
    parser.add_argument("--max-size", type=int, default=0, help="Resize source frames to match the exported COLMAP dataset.")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--erode", type=int, default=0, help="Foreground-mask erosion radius in pixels before dilation/feathering.")
    parser.add_argument("--dilate", type=int, default=0, help="Foreground-mask dilation radius in pixels before feathering.")
    parser.add_argument("--feather", type=int, default=0, help="Gaussian feather radius in pixels after dilation.")
    parser.add_argument("--premultiply-rgb", action="store_true", help="Multiply RGB by alpha so masked-out pixels are black.")
    parser.add_argument("--despill-green", action="store_true", help="Suppress green-screen color spill before writing RGBA images.")
    parser.add_argument("--despill-strength", type=float, default=1.05)
    parser.add_argument("--despill-bias", type=float, default=16.0)
    args = parser.parse_args()

    images_dir = args.dataset / "images"
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)

    frames = load_image_sequence(args.source, max_images=args.max_images, max_size=args.max_size, use_masks=True)
    written = 0
    for index, frame in enumerate(frames):
        target = images_dir / f"rgb_{index:04d}.png"
        if not target.exists():
            continue
        raw_mask = frame.mask if frame.mask is not None else np.ones(frame.image.shape[:2], dtype=np.uint8) * 255
        alpha = refine_alpha(raw_mask, erode=args.erode, dilate=args.dilate, feather=args.feather)
        rgb = frame.image
        if args.despill_green:
            rgb = despill_green_bgr(rgb, strength=args.despill_strength, bias=args.despill_bias)
        if args.premultiply_rgb:
            rgb = (rgb.astype(np.float32) * (alpha.astype(np.float32)[:, :, None] / 255.0)).round().astype(np.uint8)
        image = cv2.cvtColor(rgb, cv2.COLOR_BGR2BGRA)
        image[:, :, 3] = alpha
        if not cv2.imwrite(str(target), image):
            raise RuntimeError(f"Failed writing {target}")
        written += 1
    print(
        f"Wrote {written} RGBA masked images to {images_dir} "
        f"(erode={args.erode}, dilate={args.dilate}, feather={args.feather}, "
        f"premultiply_rgb={args.premultiply_rgb}, despill_green={args.despill_green})"
    )


if __name__ == "__main__":
    main()
