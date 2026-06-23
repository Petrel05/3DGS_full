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


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace COLMAP training images with RGBA images using foreground masks.")
    parser.add_argument("--source", required=True, help="Original image directory or video.")
    parser.add_argument("--dataset", required=True, type=Path, help="COLMAP dataset root containing images/.")
    parser.add_argument("--max-size", type=int, default=0, help="Resize source frames to match the exported COLMAP dataset.")
    parser.add_argument("--max-images", type=int, default=0)
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
        alpha = ((frame.mask if frame.mask is not None else np.ones(frame.image.shape[:2], dtype=np.uint8) * 255) > 10).astype(np.uint8) * 255
        image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2BGRA)
        image[:, :, 3] = alpha
        if not cv2.imwrite(str(target), image):
            raise RuntimeError(f"Failed writing {target}")
        written += 1
    print(f"Wrote {written} RGBA masked images to {images_dir}")


if __name__ == "__main__":
    main()
