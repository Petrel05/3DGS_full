from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop uniform letterbox borders from a COLMAP text dataset and update camera intrinsics. "
            "This is intended for VGGT padded exports before Graphdeco 3DGS training."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input COLMAP dataset root.")
    parser.add_argument("--output", required=True, type=Path, help="Output COLMAP dataset root.")
    parser.add_argument("--border-color", choices=["white", "black"], default="white")
    parser.add_argument("--threshold", type=int, default=245, help="White/black row threshold.")
    parser.add_argument("--row-fraction", type=float, default=0.99, help="Minimum border-color fraction per row.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output directory if it already exists.")
    return parser.parse_args()


def is_border_pixels(image: np.ndarray, color: str, threshold: int) -> np.ndarray:
    if color == "white":
        return np.all(image >= threshold, axis=2)
    return np.all(image <= threshold, axis=2)


def detect_vertical_crop(
    image_paths: list[Path],
    color: str,
    threshold: int,
    row_fraction: float,
) -> tuple[int, int, int, int]:
    top_values: list[int] = []
    bottom_values: list[int] = []
    height = 0
    width = 0
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        h, w = image.shape[:2]
        if height == 0:
            height, width = h, w
        elif (h, w) != (height, width):
            raise ValueError(f"All images must have the same size. {path} is {(w, h)}, expected {(width, height)}")

        row_fraction_values = is_border_pixels(image, color, threshold).mean(axis=1)
        top = 0
        while top < height and row_fraction_values[top] >= row_fraction:
            top += 1
        bottom = 0
        while bottom < height - top and row_fraction_values[height - 1 - bottom] >= row_fraction:
            bottom += 1
        top_values.append(top)
        bottom_values.append(bottom)

    top_crop = min(top_values)
    bottom_crop = min(bottom_values)
    if top_crop <= 0 and bottom_crop <= 0:
        raise RuntimeError("No vertical letterbox border was detected.")
    if top_crop + bottom_crop >= height:
        raise RuntimeError(f"Detected invalid crop top={top_crop}, bottom={bottom_crop}, height={height}")
    return width, height, top_crop, bottom_crop


def update_cameras_txt(path: Path, top_crop: int, bottom_crop: int, original_width: int, original_height: int) -> None:
    new_height = original_height - top_crop - bottom_crop
    lines = path.read_text(encoding="utf-8").splitlines()
    updated: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            updated.append(line)
            continue
        parts = stripped.split()
        if len(parts) < 5:
            updated.append(line)
            continue
        camera_id, model = parts[0], parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(value) for value in parts[4:]]
        if width != original_width or height != original_height:
            raise ValueError(
                f"Camera {camera_id} has size {(width, height)}, expected {(original_width, original_height)}"
            )
        if model == "PINHOLE":
            if len(params) != 4:
                raise ValueError(f"PINHOLE camera {camera_id} must have 4 params, got {len(params)}")
            params[3] -= top_crop
        elif model == "SIMPLE_PINHOLE":
            if len(params) != 3:
                raise ValueError(f"SIMPLE_PINHOLE camera {camera_id} must have 3 params, got {len(params)}")
            params[2] -= top_crop
        else:
            raise ValueError(f"Unsupported camera model {model!r}; add handling before cropping this dataset.")
        param_text = " ".join(f"{value:.9f}" for value in params)
        updated.append(f"{camera_id} {model} {width} {new_height} {param_text}")
    path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not (args.input / "images").is_dir():
        raise FileNotFoundError(args.input / "images")
    if not (args.input / "sparse" / "0" / "cameras.txt").is_file():
        raise FileNotFoundError(args.input / "sparse" / "0" / "cameras.txt")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} already exists. Use --overwrite to replace it.")
        shutil.rmtree(args.output)

    image_paths = sorted(args.input.joinpath("images").glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PNG images found in {args.input / 'images'}")

    width, height, top_crop, bottom_crop = detect_vertical_crop(
        image_paths,
        color=args.border_color,
        threshold=args.threshold,
        row_fraction=args.row_fraction,
    )

    shutil.copytree(args.input, args.output)
    output_images = sorted(args.output.joinpath("images").glob("*.png"))
    for path in output_images:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        cropped = image[top_crop : height - bottom_crop, :]
        if not cv2.imwrite(str(path), cropped):
            raise RuntimeError(f"Failed writing {path}")

    update_cameras_txt(
        args.output / "sparse" / "0" / "cameras.txt",
        top_crop=top_crop,
        bottom_crop=bottom_crop,
        original_width=width,
        original_height=height,
    )
    print(
        f"Cropped {len(output_images)} images from {width}x{height} to "
        f"{width}x{height - top_crop - bottom_crop} "
        f"(top={top_crop}, bottom={bottom_crop}); wrote {args.output}"
    )


if __name__ == "__main__":
    main()
