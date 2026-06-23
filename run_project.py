from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from reconstruct3d.ba import bundle_adjust
from reconstruct3d.gaussian import build_gaussians_optimized, write_gaussians
from reconstruct3d.io_utils import camera_matrix, ensure_dir, load_image_sequence, write_ply
from reconstruct3d.report import write_metrics, write_summary
from reconstruct3d.sfm import reconstruct
from reconstruct3d.vggt_adapter import run_vggt
from reconstruct3d.viewer import write_viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VGGT/BA/3D Gaussian assignment pipeline with OpenCV fallback.")
    parser.add_argument("--source", default="大作业数据/数据1-人体", help="Image directory or MP4 video.")
    parser.add_argument("--output", default="outputs/data1", help="Output directory.")
    parser.add_argument("--max-images", type=int, default=0, help="Maximum images/frames to use. 0 means all.")
    parser.add_argument("--max-size", type=int, default=0, help="Resize longest image side for reconstruction. 0 means original size.")
    parser.add_argument("--max-features", type=int, default=0, help="SIFT features per image. 0 means OpenCV default/unlimited.")
    parser.add_argument("--ba-points", type=int, default=0, help="Maximum points used in BA. 0 means all reconstructed points.")
    parser.add_argument("--ba-iters", type=int, default=0, help="Bundle adjustment function evaluations control. 0 lets SciPy converge by default.")
    parser.add_argument("--ba-backend", choices=["auto", "scipy", "legacy", "none"], default="scipy", help="BA optimizer backend.")
    parser.add_argument("--no-masks", action="store_true", help="Disable provided masks / green-screen foreground masks.")
    parser.add_argument("--backend", choices=["auto", "vggt", "opencv"], default="opencv", help="Initial camera/point-cloud backend.")
    parser.add_argument("--vggt-max-points", type=int, default=0, help="Maximum VGGT confidence-filtered points. 0 means all.")
    parser.add_argument("--vggt-confidence-percentile", type=float, default=0.0, help="Keep VGGT points above this confidence percentile. 0 means keep all finite points.")
    parser.add_argument("--vggt-model", default=".models/VGGT-1B", help="Local VGGT model directory or Hugging Face model id.")
    parser.add_argument("--allow-cpu-vggt", action="store_true", help="Force VGGT on CPU. Very slow for VGGT-1B.")
    parser.add_argument("--skip-vggt-sparse", action="store_true", help="Skip SIFT triangulation in VGGT mode and use dense VGGT points for viewer only.")
    parser.add_argument("--gaussian-backend", choices=["auto", "gsplat", "torch", "knn"], default="auto", help="Gaussian optimization backend.")
    parser.add_argument("--gaussian-source", choices=["auto", "ba", "vggt"], default="auto", help="Point source for Gaussian viewer. auto uses VGGT dense points when available.")
    parser.add_argument("--gaussian-max-points", type=int, default=0, help="Maximum Gaussian points to optimize/render. 0 means all.")
    parser.add_argument("--gaussian-iters", type=int, default=300, help="Gaussian photometric optimization iterations.")
    parser.add_argument("--gaussian-size", type=int, default=0, help="Longest side for Gaussian optimization target views. 0 means original size.")
    parser.add_argument("--gaussian-views", type=int, default=0, help="Maximum views used for Gaussian photometric optimization. 0 means all.")
    parser.add_argument("--outlier-percentile", type=float, default=100.0, help="Drop points farther than this spread percentile after BA. 100 keeps all finite points.")
    return parser.parse_args()


def vggt_accelerator_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    output = ensure_dir(args.output)
    backend = "opencv"
    vggt_error = None
    if args.backend in {"auto", "vggt"}:
        try:
            if not args.allow_cpu_vggt and not vggt_accelerator_available():
                raise RuntimeError("No CUDA/MPS accelerator is available for VGGT-1B. Use --allow-cpu-vggt to force slow CPU inference.")
            vggt = run_vggt(
                args.source,
                output,
                model_path=args.vggt_model,
                max_images=args.max_images,
                max_size=args.max_size,
                max_features=args.max_features,
                max_points=args.vggt_max_points,
                confidence_percentile=args.vggt_confidence_percentile,
                allow_cpu=args.allow_cpu_vggt,
                skip_sparse=args.skip_vggt_sparse,
            )
            rec = vggt.reconstruction
            raw_vggt_points = vggt.raw_points
            raw_vggt_colors = vggt.raw_colors
            backend = vggt.backend
            frames = vggt.frames
            h, w = frames[0].image.shape[:2]
        except Exception as exc:
            if args.backend == "vggt":
                raise
            vggt_error = f"{type(exc).__name__}: {exc}"
            frames = load_image_sequence(args.source, max_images=args.max_images, max_size=args.max_size, use_masks=not args.no_masks)
            h, w = frames[0].image.shape[:2]
            k = camera_matrix(w, h)
            rec = reconstruct(frames, k, max_features=args.max_features)
            raw_vggt_points = None
            raw_vggt_colors = None
            backend = "OpenCV SfM fallback"
    else:
        frames = load_image_sequence(args.source, max_images=args.max_images, max_size=args.max_size, use_masks=not args.no_masks)
        h, w = frames[0].image.shape[:2]
        k = camera_matrix(w, h)
        rec = reconstruct(frames, k, max_features=args.max_features)
        raw_vggt_points = None
        raw_vggt_colors = None
    if len(rec.points) == 0 and raw_vggt_points is None:
        raise RuntimeError("No valid 3D points were reconstructed. Try increasing --max-size or --max-features.")

    if args.ba_backend != "none" and rec.tracks and any(len(track) >= 2 for track in rec.tracks):
        ba = bundle_adjust(
            rec.camera_matrix,
            rec.rotations,
            rec.translations,
            rec.points,
            rec.tracks,
            max_points=args.ba_points,
            max_iterations=args.ba_iters,
            backend="opencv" if args.ba_backend == "legacy" else args.ba_backend,
        )
        points_for_output = ba.points
        rotations_for_output = ba.rotations
        translations_for_output = ba.translations
    else:
        ba = None
        points_for_output = rec.points
        rotations_for_output = rec.rotations
        translations_for_output = rec.translations

    finite = np.isfinite(points_for_output).all(axis=1) if len(points_for_output) else np.zeros((0,), dtype=bool)
    points = points_for_output[finite]
    colors = rec.colors[finite] if len(rec.colors) else np.zeros((0, 3), dtype=np.float64)
    if len(points) > 0 and args.outlier_percentile < 100.0:
        spread = np.linalg.norm(points - np.median(points, axis=0), axis=1)
        keep = spread < np.percentile(spread, args.outlier_percentile)
        points = points[keep]
        colors = colors[keep]

    ply_path = output / "point_cloud.ply"
    vggt_ply_path = output / "vggt_initial_point_cloud.ply"
    json_path = output / "gaussians.json"
    html_path = output / "viewer.html"
    metrics_path = output / "metrics.json"
    summary_path = output / "summary.md"

    write_ply(ply_path, points, colors)
    gaussian_points_input = points
    gaussian_colors_input = colors
    gaussian_source = "ba"
    if raw_vggt_points is not None and raw_vggt_colors is not None:
        finite_raw = np.isfinite(raw_vggt_points).all(axis=1)
        raw_points = raw_vggt_points[finite_raw]
        raw_colors = raw_vggt_colors[finite_raw]
        if args.vggt_max_points > 0 and len(raw_points) > args.vggt_max_points:
            idx = np.linspace(0, len(raw_points) - 1, args.vggt_max_points, dtype=int)
            raw_points = raw_points[idx]
            raw_colors = raw_colors[idx]
        write_ply(vggt_ply_path, raw_points, raw_colors)
        if args.gaussian_source in {"auto", "vggt"} and len(raw_points) > 0:
            gaussian_points_input = raw_points
            gaussian_colors_input = raw_colors
            gaussian_source = "vggt"
    gaussian_result = build_gaussians_optimized(
        gaussian_points_input,
        gaussian_colors_input,
        frames=frames,
        camera_matrix=rec.camera_matrix,
        rotations=rotations_for_output,
        translations=translations_for_output,
        max_points=args.gaussian_max_points,
        backend=args.gaussian_backend,
        iterations=args.gaussian_iters,
        image_size=args.gaussian_size,
        max_views=args.gaussian_views,
    )
    gaussians = gaussian_result.data
    write_gaussians(json_path, gaussians)
    write_viewer(html_path, gaussians)

    metrics = {
        "source": str(args.source),
        "backend": backend,
        "vggt_error": vggt_error,
        "images": len(frames),
        "image_size": [int(w), int(h)],
        "use_masks": not args.no_masks,
        "points_before_ba": int(len(rec.points)),
        "points_after_ba": int(len(points)),
        "gaussian_source": gaussian_source,
        "gaussian_input_points": int(len(gaussian_points_input)),
        "gaussian_points": int(len(gaussians["points"])),
        "initial_reprojection_rmse": float(rec.reprojection_rmse) if np.isfinite(rec.reprojection_rmse) else None,
        "ba_initial_rmse": float(ba.initial_rmse) if ba else None,
        "ba_final_rmse": float(ba.final_rmse) if ba else None,
        "ba_iterations": int(ba.iterations) if ba else 0,
        "ba_accepted_steps": int(ba.accepted_steps) if ba else 0,
        "ba_backend": ba.backend if ba else None,
        "gaussian_backend": gaussian_result.backend,
        "gaussian_initial_loss": gaussian_result.initial_loss,
        "gaussian_final_loss": gaussian_result.final_loss,
        "gaussian_iterations": gaussian_result.iterations,
        "limits": {
            "max_images": int(args.max_images),
            "max_size": int(args.max_size),
            "max_features": int(args.max_features),
            "ba_points": int(args.ba_points),
            "vggt_max_points": int(args.vggt_max_points),
            "gaussian_max_points": int(args.gaussian_max_points),
            "gaussian_size": int(args.gaussian_size),
            "gaussian_views": int(args.gaussian_views),
            "outlier_percentile": float(args.outlier_percentile),
        },
        "pair_inliers": [int(v) for v in rec.pair_inliers],
        "outputs": {
            "ply": str(ply_path),
            "vggt_initial_ply": str(vggt_ply_path) if raw_vggt_points is not None else None,
            "gaussians": str(json_path),
            "viewer": str(html_path),
            "summary": str(summary_path),
        },
    }
    write_metrics(metrics_path, metrics)
    write_summary(summary_path, metrics)
    print(f"Wrote {html_path}")
    if ba:
        print(f"Final BA RMSE: {metrics['ba_final_rmse']:.3f} px, points: {metrics['points_after_ba']}")
    else:
        print(f"Backend: {backend}, points: {metrics['points_after_ba']} (BA skipped: VGGT dense points have no 2D tracks)")


if __name__ == "__main__":
    main()
