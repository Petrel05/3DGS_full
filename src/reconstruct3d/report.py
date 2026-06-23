from __future__ import annotations

import json
from pathlib import Path


def write_metrics(path: str | Path, metrics: dict) -> None:
    Path(path).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(path: str | Path, metrics: dict) -> None:
    ba_initial = metrics.get("ba_initial_rmse")
    ba_final = metrics.get("ba_final_rmse")
    initial_reprojection = metrics.get("initial_reprojection_rmse")
    gaussian_initial = metrics.get("gaussian_initial_loss")
    gaussian_final = metrics.get("gaussian_final_loss")
    initial_reprojection_text = f"{initial_reprojection:.3f} px" if initial_reprojection is not None else "N/A"
    ba_initial_text = f"{ba_initial:.3f} px" if ba_initial is not None else "N/A"
    ba_final_text = f"{ba_final:.3f} px" if ba_final is not None else "N/A"
    gaussian_initial_text = f"{gaussian_initial:.6f}" if gaussian_initial is not None else "N/A"
    gaussian_final_text = f"{gaussian_final:.6f}" if gaussian_final is not None else "N/A"
    limits = metrics.get("limits", {})
    text = f"""# Reconstruction Run Summary

- Input: `{metrics['source']}`
- Backend: {metrics.get('backend', 'unknown')}
- Images used: {metrics['images']}
- Sparse points before BA: {metrics['points_before_ba']}
- Sparse points after BA/filtering: {metrics['points_after_ba']}
- Gaussian source: {metrics.get('gaussian_source', 'N/A')}
- Gaussian input/rendered points: {metrics.get('gaussian_input_points', 'N/A')} -> {metrics.get('gaussian_points', 'N/A')}
- Initial reprojection RMSE: {initial_reprojection_text}
- BA backend: {metrics.get('ba_backend', 'N/A')}
- BA initial RMSE: {ba_initial_text}
- BA final RMSE: {ba_final_text}
- BA accepted steps: {metrics['ba_accepted_steps']}
- Gaussian backend: {metrics.get('gaussian_backend', 'N/A')}
- Gaussian photometric loss: {gaussian_initial_text} -> {gaussian_final_text}
- Limits: max_images={limits.get('max_images', 'N/A')}, max_size={limits.get('max_size', 'N/A')}, max_features={limits.get('max_features', 'N/A')}, ba_points={limits.get('ba_points', 'N/A')}, gaussian_max_points={limits.get('gaussian_max_points', 'N/A')}, gaussian_size={limits.get('gaussian_size', 'N/A')}, gaussian_views={limits.get('gaussian_views', 'N/A')}
- Pair pose inliers: {metrics['pair_inliers']}

Generated files:

- `point_cloud.ply`
- `gaussians.json`
- `viewer.html`
- `metrics.json`
"""
    Path(path).write_text(text, encoding="utf-8")
