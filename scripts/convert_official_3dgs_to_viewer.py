from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reconstruct3d.gaussian import normalize_points_with_scale
from reconstruct3d.viewer import write_viewer


SH_C0 = 0.28209479177387814


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_official_gaussians(path: Path, radius_multiplier: float, max_radius: float) -> tuple[dict, dict]:
    ply = PlyData.read(path)
    vertices = ply["vertex"]
    names = vertices.data.dtype.names or ()
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2"}
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError(f"{path} is missing required official 3DGS fields: {', '.join(missing)}")

    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64)
    points, scene_scale = normalize_points_with_scale(points)

    sh_dc = np.column_stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]).astype(np.float64)
    colors = np.clip(sh_dc * SH_C0 + 0.5, 0.0, 1.0)

    opacities = np.clip(sigmoid(np.asarray(vertices["opacity"], dtype=np.float64)), 0.02, 1.0)

    log_scales = np.column_stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]]).astype(np.float64)
    radii = np.exp(np.mean(log_scales, axis=1)) / max(scene_scale, 1e-8)
    radii = np.clip(radii * radius_multiplier, 0.0012, max_radius)

    gaussians = {
        "points": points.round(6).tolist(),
        "colors": colors.round(4).tolist(),
        "radii": radii.round(6).tolist(),
        "opacities": opacities.round(4).tolist(),
        "viewer_point_scale": 0.55,
        "note": (
            "Converted from official Graphdeco 3D Gaussian Splatting PLY. "
            "Colors use SH DC terms, opacities use sigmoid logits, and log-scales are mapped to circular WebGL splats."
        ),
    }
    metrics = {
        "source_ply": str(path),
        "points": int(len(points)),
        "scene_scale": float(scene_scale),
        "radius_multiplier": float(radius_multiplier),
        "max_radius": float(max_radius),
        "radius_min": float(radii.min()) if len(radii) else None,
        "radius_median": float(np.median(radii)) if len(radii) else None,
        "radius_max": float(radii.max()) if len(radii) else None,
        "opacity_min": float(opacities.min()) if len(opacities) else None,
        "opacity_median": float(np.median(opacities)) if len(opacities) else None,
        "opacity_max": float(opacities.max()) if len(opacities) else None,
    }
    return gaussians, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official Graphdeco 3DGS PLY to this project's WebGL viewer.")
    parser.add_argument("--input", required=True, type=Path, help="Official point_cloud.ply")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for viewer.html and gaussians.json")
    parser.add_argument("--radius-multiplier", type=float, default=2.5, help="Scale circular viewer splats.")
    parser.add_argument("--max-radius", type=float, default=0.035, help="Clamp large viewer splats after normalization.")
    args = parser.parse_args()

    gaussians, metrics = load_official_gaussians(args.input, args.radius_multiplier, args.max_radius)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "gaussians.json").write_text(json.dumps(gaussians, ensure_ascii=False), encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_viewer(args.output / "viewer.html", gaussians)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
