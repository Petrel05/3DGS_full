from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLED_VIEWER_BIN = ROOT / "SIBR" / "viewer" / "bin" / "SIBR_gaussianViewer_app.exe"

DATASETS = {
    "data1": {
        "model": ROOT / "outputs" / "official_data1_graphdeco_30k_masked_clean_bg",
        "source": ROOT / "outputs" / "official_data1_colmap_50k_masked_clean",
    },
    "data2": {
        "model": ROOT / "outputs" / "official_data2_graphdeco_30k_masked_clean_bg",
        "source": ROOT / "outputs" / "official_data2_colmap_50k_masked_clean",
    },
    "scene128": {
        "model": ROOT / "outputs" / "official_scene128_graphdeco_30k_cropped",
        "source": ROOT / "outputs" / "official_scene128_colmap_50k_cropped",
    },
}


def default_viewer_bin() -> Path | None:
    if os.environ.get("SIBR_VIEWER_BIN"):
        return Path(os.environ["SIBR_VIEWER_BIN"])
    if BUNDLED_VIEWER_BIN.is_file():
        return BUNDLED_VIEWER_BIN
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the official Graphdeco SIBR Gaussian viewer for one of this project's final outputs. "
            "The current Linux machine cannot run the bundled Windows SIBR executable. "
            "Use --dry-run here, or run this on a machine where the selected SIBR binary works."
        )
    )
    parser.add_argument("dataset", choices=sorted(DATASETS), help="Final output to view.")
    parser.add_argument(
        "--viewer-bin",
        type=Path,
        default=default_viewer_bin(),
        help=(
            "Path to SIBR_gaussianViewer_app. Defaults to SIBR_VIEWER_BIN, then "
            "SIBR/viewer/bin/SIBR_gaussianViewer_app.exe when present."
        ),
    )
    parser.add_argument("--model", type=Path, help="Override Graphdeco model/output directory.")
    parser.add_argument("--source", type=Path, help="Override COLMAP source dataset directory.")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration to load in the SIBR viewer.")
    parser.add_argument("--rendering-size", nargs=2, metavar=("WIDTH", "HEIGHT"), help="Optional SIBR rendering size.")
    parser.add_argument("--no-source", action="store_true", help="Do not pass -s/--source-path to SIBR.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing it.")
    args, viewer_args = parser.parse_known_args()
    if viewer_args and viewer_args[0] == "--":
        viewer_args = viewer_args[1:]
    args.viewer_args = viewer_args
    return args


def require_path(path: Path, description: str, must_be_file: bool = False) -> Path:
    path = path.expanduser().resolve()
    exists = path.is_file() if must_be_file else path.exists()
    if not exists:
        kind = "file" if must_be_file else "path"
        raise FileNotFoundError(f"{description} {kind} does not exist: {path}")
    return path


def validate_source_dataset(source: Path) -> None:
    require_path(source / "images", "COLMAP images directory")
    sparse = require_path(source / "sparse" / "0", "COLMAP sparse/0 directory")
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        require_path(sparse / name, f"COLMAP {name}", must_be_file=True)
    if not any((source / "images").iterdir()):
        raise FileNotFoundError(f"COLMAP images directory is empty: {source / 'images'}")


def main() -> None:
    args = parse_args()
    defaults = DATASETS[args.dataset]

    if args.viewer_bin is None:
        raise SystemExit("Missing SIBR viewer binary. Pass --viewer-bin or set SIBR_VIEWER_BIN.")

    viewer_bin = require_path(args.viewer_bin, "SIBR viewer binary", must_be_file=True)
    model = require_path(args.model or defaults["model"], "Graphdeco model directory")
    source = None
    if not args.no_source:
        source = require_path(args.source or defaults["source"], "COLMAP source dataset")
        validate_source_dataset(source)

    point_cloud = model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    require_path(point_cloud, "Graphdeco point cloud", must_be_file=True)

    command = [str(viewer_bin), "-m", str(model), "--iteration", str(args.iteration)]
    if source is not None:
        command.extend(["-s", str(source)])
    if args.rendering_size:
        command.extend(["--rendering-size", args.rendering_size[0], args.rendering_size[1]])
    if args.viewer_args:
        passthrough = args.viewer_args[1:] if args.viewer_args[0] == "--" else args.viewer_args
        command.extend(passthrough)

    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
