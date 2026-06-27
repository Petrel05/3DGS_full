from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from reconstruct3d.ba import bundle_adjust
from reconstruct3d.gaussian import build_gaussians_optimized, write_gaussians
from reconstruct3d.io_utils import camera_matrix, ensure_dir, load_image_sequence, write_colmap_text_dataset, write_ply
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
    parser.add_argument("--vggt-preprocess-mode", choices=["pad", "crop"], default="pad", help="VGGT image preprocessing mode passed to load_and_preprocess_images.")
    parser.add_argument(
        "--vggt-mask-background",
        choices=["none", "black", "gray", "blur"],
        default="none",
        help="Replace mask-out background before VGGT inference. This is a test-time input optimization for masked subjects.",
    )
    parser.add_argument("--vggt-mask-erode", type=int, default=0, help="Erode the foreground mask before VGGT background replacement.")
    parser.add_argument("--allow-cpu-vggt", action="store_true", help="Force VGGT on CPU. Very slow for VGGT-1B.")
    parser.add_argument("--skip-vggt-sparse", action="store_true", help="Skip SIFT triangulation in VGGT mode and use dense VGGT points for viewer only.")
    parser.add_argument("--vggt-sparse-match-window", type=int, default=1, help="Adjacent image window for VGGT-initialized sparse tracks.")
    parser.add_argument("--vggt-sparse-loop-closure", action="store_true", help="Also match the end of an ordered turntable sequence back to the beginning.")
    parser.add_argument("--gaussian-backend", choices=["auto", "gsplat", "torch", "knn"], default="auto", help="Gaussian optimization backend.")
    parser.add_argument("--gaussian-source", choices=["auto", "ba", "vggt"], default="auto", help="Point source for Gaussian viewer. auto uses VGGT dense points when available.")
    parser.add_argument("--gaussian-max-points", type=int, default=0, help="Maximum Gaussian points to optimize/render. 0 means all.")
    parser.add_argument("--gaussian-iters", type=int, default=300, help="Gaussian photometric optimization iterations.")
    parser.add_argument("--gaussian-size", type=int, default=0, help="Longest side for Gaussian optimization target views. 0 means original size.")
    parser.add_argument("--gaussian-views", type=int, default=0, help="Maximum views used for Gaussian photometric optimization. 0 means all.")
    parser.add_argument("--cuda-memory-guard-fraction", type=float, default=0.92, help="Stop gsplat early and save current Gaussians when CUDA memory use exceeds this fraction. 0 disables.")
    parser.add_argument("--dense-camera-filter", choices=["none", "visible", "mask"], default="none", help="Filter VGGT dense points with optimized cameras before Gaussian optimization.")
    parser.add_argument("--dense-min-visible-views", type=int, default=1, help="Minimum camera views that must see a dense VGGT point.")
    parser.add_argument("--export-colmap-dataset", default="", help="Optional COLMAP text dataset path for official Graphdeco 3DGS training.")
    parser.add_argument("--export-colmap-alpha-mask", action="store_true", help="Write COLMAP images as RGBA PNGs using foreground masks as alpha.")
    parser.add_argument("--outlier-percentile", type=float, default=100.0, help="Drop points farther than this spread percentile after BA. 100 keeps all finite points.")
    return parser.parse_args()


def vggt_accelerator_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def filter_dense_points_by_cameras(
    points: np.ndarray,
    colors: np.ndarray,
    frames,
    camera_matrix: np.ndarray,
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    mode: str,
    min_visible_views: int,
    chunk_size: int = 50000,
) -> tuple[np.ndarray, np.ndarray, int]:
    if mode == "none" or len(points) == 0:
        return points, colors, int(len(points))
    min_visible_views = max(1, int(min_visible_views))
    keep_parts = []
    color_parts = []
    for start in range(0, len(points), chunk_size):
        pts = points[start : start + chunk_size]
        visible = np.zeros((len(pts),), dtype=np.int16)
        for frame, rot, trans in zip(frames, rotations, translations):
            h, w = frame.image.shape[:2]
            cam = (rot @ pts.T).T + trans.reshape(1, 3)
            z = cam[:, 2]
            valid = z > 1e-5
            pix = (camera_matrix @ cam.T).T
            xy = pix[:, :2] / np.maximum(pix[:, 2:3], 1e-8)
            x = np.rint(xy[:, 0]).astype(np.int64)
            y = np.rint(xy[:, 1]).astype(np.int64)
            valid &= (x >= 0) & (x < w) & (y >= 0) & (y < h)
            if mode == "mask" and frame.mask is not None and valid.any():
                inside = np.zeros_like(valid)
                idx = np.flatnonzero(valid)
                inside[idx] = frame.mask[y[idx], x[idx]] > 10
                valid = inside
            visible += valid.astype(np.int16)
        keep = visible >= min_visible_views
        if keep.any():
            keep_parts.append(pts[keep])
            color_parts.append(colors[start : start + chunk_size][keep])
    if not keep_parts:
        return points[:0], colors[:0], 0
    return np.vstack(keep_parts), np.vstack(color_parts), int(sum(len(part) for part in keep_parts))


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
                mode=args.vggt_preprocess_mode,
                input_background_mode=args.vggt_mask_background,
                input_mask_erode=args.vggt_mask_erode,
                allow_cpu=args.allow_cpu_vggt,
                skip_sparse=args.skip_vggt_sparse,
                sparse_match_window=args.vggt_sparse_match_window,
                sparse_loop_closure=args.vggt_sparse_loop_closure,
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
    dense_points_before_filter = None
    dense_points_after_filter = None
    if raw_vggt_points is not None and raw_vggt_colors is not None:
        finite_raw = np.isfinite(raw_vggt_points).all(axis=1)
        raw_points = raw_vggt_points[finite_raw]
        raw_colors = raw_vggt_colors[finite_raw]
        if args.vggt_max_points > 0 and len(raw_points) > args.vggt_max_points:
            idx = np.linspace(0, len(raw_points) - 1, args.vggt_max_points, dtype=int)
            raw_points = raw_points[idx]
            raw_colors = raw_colors[idx]
        write_ply(vggt_ply_path, raw_points, raw_colors)
        dense_points_before_filter = int(len(raw_points))
        filtered_raw_points, filtered_raw_colors, dense_visible = filter_dense_points_by_cameras(
            raw_points,
            raw_colors,
            frames,
            rec.camera_matrix,
            rotations_for_output,
            translations_for_output,
            mode=args.dense_camera_filter,
            min_visible_views=args.dense_min_visible_views,
        )
        dense_points_after_filter = dense_visible
        if args.gaussian_source in {"auto", "vggt"} and len(raw_points) > 0:
            if args.dense_camera_filter != "none" and len(filtered_raw_points) > 0:
                gaussian_points_input = filtered_raw_points
                gaussian_colors_input = filtered_raw_colors
            else:
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
        cuda_memory_guard_fraction=args.cuda_memory_guard_fraction,
    )
    colmap_dataset_path = None
    if args.export_colmap_dataset:
        colmap_dataset_path = write_colmap_text_dataset(
            args.export_colmap_dataset,
            frames,
            rec.camera_matrix,
            rotations_for_output,
            translations_for_output,
            gaussian_points_input,
            gaussian_colors_input,
            alpha_mask_images=args.export_colmap_alpha_mask,
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
        "dense_points_before_filter": dense_points_before_filter,
        "dense_points_after_filter": dense_points_after_filter,
        "dense_camera_filter": args.dense_camera_filter,
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
            "cuda_memory_guard_fraction": float(args.cuda_memory_guard_fraction),
            "vggt_sparse_match_window": int(args.vggt_sparse_match_window),
            "vggt_sparse_loop_closure": bool(args.vggt_sparse_loop_closure),
            "vggt_preprocess_mode": args.vggt_preprocess_mode,
            "vggt_mask_background": args.vggt_mask_background,
            "vggt_mask_erode": int(args.vggt_mask_erode),
            "dense_min_visible_views": int(args.dense_min_visible_views),
            "export_colmap_alpha_mask": bool(args.export_colmap_alpha_mask),
            "outlier_percentile": float(args.outlier_percentile),
        },
        "pair_inliers": [int(v) for v in rec.pair_inliers],
        "outputs": {
            "ply": str(ply_path),
            "vggt_initial_ply": str(vggt_ply_path) if raw_vggt_points is not None else None,
            "gaussians": str(json_path),
            "viewer": str(html_path),
            "summary": str(summary_path),
            "colmap_dataset": str(colmap_dataset_path) if colmap_dataset_path is not None else None,
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
