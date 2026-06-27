from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA2 = ROOT / "大作业数据" / "数据2-人体"
OUTPUT_ROOT = ROOT / "outputs" / "vggt_data2_network_experiments"


EXPERIMENTS = [
    {
        "name": "baseline_p25",
        "description": "VGGT baseline: percentile 25, pad preprocessing.",
        "source": DATA2,
        "confidence": 25,
        "preprocess_mode": "pad",
        "mask_background": "none",
        "mask_erode": 0,
        "match_window": 1,
        "loop_closure": False,
        "ba_points": 120,
    },
    {
        "name": "p0_strict_points",
        "description": "Keep all finite VGGT confidence points, then use strict foreground filtering.",
        "source": DATA2,
        "confidence": 0,
        "preprocess_mode": "pad",
        "mask_background": "none",
        "mask_erode": 0,
        "match_window": 1,
        "loop_closure": False,
        "ba_points": 120,
    },
    {
        "name": "masked_gray_input_p0",
        "description": "Segmentation-guided VGGT input: replace background with neutral gray before inference.",
        "source": DATA2,
        "confidence": 0,
        "preprocess_mode": "pad",
        "mask_background": "gray",
        "mask_erode": 1,
        "match_window": 1,
        "loop_closure": False,
        "ba_points": 120,
    },
    {
        "name": "foreground_crop_p0",
        "description": "Foreground-centric tight square crop before VGGT to increase subject resolution.",
        "source": OUTPUT_ROOT / "data2_foreground_crop",
        "prepare_crop": True,
        "confidence": 0,
        "preprocess_mode": "pad",
        "mask_background": "none",
        "mask_erode": 0,
        "match_window": 1,
        "loop_closure": False,
        "ba_points": 120,
    },
    {
        "name": "wide_tracks_loop_p0",
        "description": "Use all VGGT points and strengthen sparse tracks with match window 2 + loop closure.",
        "source": DATA2,
        "confidence": 0,
        "preprocess_mode": "pad",
        "mask_background": "none",
        "mask_erode": 0,
        "match_window": 2,
        "loop_closure": True,
        "ba_points": 240,
    },
]


def run(command: list[str], dry_run: bool = False) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def natural_key(path: Path) -> list[object]:
    import re

    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def mask_path_for(image_path: Path) -> Path:
    return image_path.with_name(image_path.name.replace("rgb_", "msk_", 1))


def prepare_foreground_crop(source: Path, output: Path, padding_ratio: float = 0.18) -> None:
    if output.exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    image_paths = sorted([p for p in source.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.name.startswith("rgb_")], key=natural_key)
    for image_path in image_paths:
        mask_path = mask_path_for(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Missing image or mask for {image_path}")
        ys, xs = np.where(mask > 10)
        if len(xs) == 0:
            raise RuntimeError(f"Empty foreground mask: {mask_path}")
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        side = max(x1 - x0, y1 - y0)
        side = int(round(side * (1.0 + 2.0 * padding_ratio)))
        h, w = image.shape[:2]
        x0 = max(0, int(round(cx - side * 0.5)))
        y0 = max(0, int(round(cy - side * 0.5)))
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)
        x0 = max(0, x1 - side)
        y0 = max(0, y1 - side)
        cv2.imwrite(str(output / image_path.name), image[y0:y1, x0:x1])
        cv2.imwrite(str(output / mask_path.name), mask[y0:y1, x0:x1])


def count_points3d(dataset: Path) -> int:
    points_path = dataset / "sparse" / "0" / "points3D.txt"
    if not points_path.exists():
        return 0
    return sum(1 for line in points_path.read_text().splitlines() if line and not line.startswith("#"))


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_experiment(exp: dict, args: argparse.Namespace) -> dict:
    if exp.get("prepare_crop"):
        prepare_foreground_crop(DATA2, Path(exp["source"]))

    exp_dir = OUTPUT_ROOT / exp["name"]
    export_dataset = OUTPUT_ROOT / f"{exp['name']}_colmap"
    masked_dataset = OUTPUT_ROOT / f"{exp['name']}_colmap_masked_strict"
    filtered_dataset = OUTPUT_ROOT / f"{exp['name']}_colmap_masked_strict_fg"

    if not (exp_dir / "metrics.json").exists() or args.force:
        if exp_dir.exists() and args.force:
            shutil.rmtree(exp_dir)
        if export_dataset.exists() and args.force:
            shutil.rmtree(export_dataset)
        command = [
            sys.executable,
            "run_project.py",
            "--backend",
            "vggt",
            "--vggt-model",
            ".models/VGGT-1B",
            "--source",
            str(exp["source"]),
            "--output",
            str(exp_dir),
            "--max-images",
            "0",
            "--ba-backend",
            "scipy",
            "--ba-points",
            str(exp["ba_points"]),
            "--vggt-sparse-match-window",
            str(exp["match_window"]),
            "--vggt-confidence-percentile",
            str(exp["confidence"]),
            "--vggt-max-points",
            "50000",
            "--vggt-preprocess-mode",
            exp["preprocess_mode"],
            "--vggt-mask-background",
            exp["mask_background"],
            "--vggt-mask-erode",
            str(exp["mask_erode"]),
            "--gaussian-source",
            "vggt",
            "--dense-camera-filter",
            "mask",
            "--gaussian-backend",
            "knn",
            "--gaussian-max-points",
            "1000",
            "--export-colmap-dataset",
            str(export_dataset),
            "--outlier-percentile",
            "100",
        ]
        if exp["loop_closure"]:
            command.append("--vggt-sparse-loop-closure")
        run(command, dry_run=args.dry_run)

    if not args.dry_run:
        if masked_dataset.exists() and args.force:
            shutil.rmtree(masked_dataset)
        if not masked_dataset.exists():
            shutil.copytree(export_dataset, masked_dataset, copy_function=shutil.copy2)
            source_for_masks = DATA2 if not exp.get("prepare_crop") else Path(exp["source"])
            run(
                [
                    sys.executable,
                    "scripts/apply_alpha_masks_to_colmap_images.py",
                    "--source",
                    str(source_for_masks),
                    "--dataset",
                    str(masked_dataset),
                    "--max-size",
                    "518",
                    "--erode",
                    "2",
                    "--feather",
                    "1",
                    "--premultiply-rgb",
                    "--despill-green",
                ]
            )

        if filtered_dataset.exists() and args.force:
            shutil.rmtree(filtered_dataset)
        if not filtered_dataset.exists():
            run(
                [
                    sys.executable,
                    "scripts/filter_colmap_points_by_alpha.py",
                    "--dataset",
                    str(masked_dataset),
                    "--output-dataset",
                    str(filtered_dataset),
                    "--alpha-threshold",
                    "96",
                    "--min-foreground-views",
                    "3",
                    "--min-foreground-ratio",
                    "0.70",
                ]
            )

    metrics = load_metrics(exp_dir / "metrics.json") if not args.dry_run else {}
    return {
        "name": exp["name"],
        "description": exp["description"],
        "output": str(exp_dir),
        "colmap": str(export_dataset),
        "strict_colmap": str(filtered_dataset),
        "initial_rmse": metrics.get("initial_reprojection_rmse"),
        "ba_initial_rmse": metrics.get("ba_initial_rmse"),
        "ba_final_rmse": metrics.get("ba_final_rmse"),
        "sparse_points": metrics.get("points_after_ba"),
        "dense_input_points": metrics.get("gaussian_input_points"),
        "strict_foreground_points": count_points3d(filtered_dataset) if not args.dry_run else None,
        "pair_inliers_sum": int(sum(metrics.get("pair_inliers", []))) if metrics else None,
    }


def write_summary(rows: list[dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "vggt_network_optimization_data2_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Data2 VGGT-side Optimization Experiments",
        "",
        "The experiments compare test-time VGGT-side changes before Graphdeco training.",
        "",
        "| Experiment | Method | Initial RMSE | BA final RMSE | Sparse points | Pair inliers | Strict foreground points |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {description} | {initial_rmse:.3f} | {ba_final_rmse:.3f} | {sparse_points} | {pair_inliers_sum} | {strict_foreground_points} |".format(
                **{
                    **row,
                    "initial_rmse": row["initial_rmse"] if row["initial_rmse"] is not None else float("nan"),
                    "ba_final_rmse": row["ba_final_rmse"] if row["ba_final_rmse"] is not None else float("nan"),
                }
            )
        )
    lines.extend(
        [
            "",
            "Recommended next step: train Graphdeco 3DGS on the best strict foreground COLMAP dataset and compare SIBR artifacts.",
        ]
    )
    (OUTPUT_ROOT / "vggt_network_optimization_data2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=[], help="Optional experiment names to run.")
    parser.add_argument("--force", action="store_true", help="Recompute existing outputs.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected = [exp for exp in EXPERIMENTS if not args.only or exp["name"] in set(args.only)]
    rows = [run_experiment(exp, args) for exp in selected]
    if not args.dry_run:
        write_summary(rows)
        print(OUTPUT_ROOT / "vggt_network_optimization_data2_summary.md")


if __name__ == "__main__":
    main()
