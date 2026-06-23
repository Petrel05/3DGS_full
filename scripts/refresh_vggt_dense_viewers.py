from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reconstruct3d.gaussian import build_gaussians
from reconstruct3d.viewer import write_viewer


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    count = 0
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
        if line == "end_header":
            start = i + 1
            break
    points = []
    colors_bgr = []
    for line in lines[start : start + count]:
        vals = line.split()
        if len(vals) < 6:
            continue
        points.append([float(vals[0]), float(vals[1]), float(vals[2])])
        red, green, blue = int(vals[3]), int(vals[4]), int(vals[5])
        colors_bgr.append([blue, green, red])
    return np.asarray(points, dtype=np.float64), np.asarray(colors_bgr, dtype=np.float64)


def refresh(output_dir: Path) -> None:
    ply = output_dir / "vggt_initial_point_cloud.ply"
    metrics_path = output_dir / "metrics.json"
    if not ply.exists():
        raise FileNotFoundError(ply)
    points, colors = read_ply(ply)
    gaussians = build_gaussians(points, colors, max_points=0)
    (output_dir / "gaussians.json").write_text(json.dumps(gaussians, ensure_ascii=False, indent=2), encoding="utf-8")
    write_viewer(output_dir / "viewer.html", gaussians)
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["gaussian_source"] = "vggt_initial_point_cloud.ply"
        metrics["gaussian_input_points"] = int(len(points))
        metrics["gaussian_points"] = int(len(gaussians["points"]))
        metrics["gaussian_backend"] = "knn-density-from-vggt-initial"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"refreshed {output_dir}: {len(points)} input points -> {len(gaussians['points'])} gaussian points")


def main() -> None:
    targets = sys.argv[1:] or ["outputs/vggt_data1", "outputs/vggt_data2", "outputs/vggt_scene"]
    for target in targets:
        refresh(ROOT / target)


if __name__ == "__main__":
    main()
