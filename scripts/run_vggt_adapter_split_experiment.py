from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from reconstruct3d.ba import bundle_adjust
from reconstruct3d.io_utils import Frame, ensure_dir, write_colmap_text_dataset
from reconstruct3d.sfm import Observation, project_points, reconstruct_with_known_cameras
from reconstruct3d.vggt_adapter import run_vggt
from run_project import filter_dense_points_by_cameras
from scripts.filter_colmap_points_by_alpha import qvec_to_rotmat, read_cameras, read_images, read_points


DATA2 = ROOT / "大作业数据" / "数据2-人体"
GRAPHDECO_ROOT = Path("/tmp/gaussian-splatting-official")
OUTPUT_ROOT = ROOT / "outputs" / "vggt_adapter_split_data2"
PPT_ASSETS = ROOT / "ppt_assets"
DEFAULT_REUSE_COLMAP = ROOT / "outputs" / "vggt_data2_network_experiments" / "p0_strict_points_colmap_masked_strict"


@dataclass
class FoldResult:
    candidate: str
    fold: int
    train_views: list[int]
    val_views: list[int]
    train_rmse_before: float
    train_rmse_after: float
    val_rmse_before: float | None
    val_rmse_after: float | None
    optimized_tracks: int
    backend: str


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def compact_tracks(
    points: np.ndarray,
    tracks: list[list[Observation]],
    allowed_views: list[int],
) -> tuple[np.ndarray, list[list[Observation]], list[int], dict[int, int]]:
    allowed = set(allowed_views)
    view_map = {view_id: compact_id for compact_id, view_id in enumerate(allowed_views)}
    kept_points = []
    kept_tracks = []
    kept_indices = []
    for point_id, (point, track) in enumerate(zip(points, tracks)):
        compact_track = [
            Observation(image_id=view_map[obs.image_id], xy=obs.xy.copy())
            for obs in track
            if obs.image_id in allowed
        ]
        if len(compact_track) >= 2:
            kept_points.append(point)
            kept_tracks.append(compact_track)
            kept_indices.append(point_id)
    if not kept_points:
        return np.zeros((0, 3), dtype=np.float64), [], [], view_map
    return np.vstack(kept_points).astype(np.float64), kept_tracks, kept_indices, view_map


def reprojection_rmse(
    k: np.ndarray,
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    points: np.ndarray,
    tracks: list[list[Observation]],
    allowed_views: set[int] | None = None,
) -> float | None:
    residuals = []
    for point, track in zip(points, tracks):
        point2 = point.reshape(1, 3)
        for obs in track:
            if allowed_views is not None and obs.image_id not in allowed_views:
                continue
            xy = project_points(point2, k, rotations[obs.image_id], translations[obs.image_id])[0]
            residuals.extend(xy - obs.xy)
    if not residuals:
        return None
    arr = np.asarray(residuals, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)))


def validation_rmse_for_kept_tracks(
    k: np.ndarray,
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    optimized_points: np.ndarray,
    original_tracks: list[list[Observation]],
    kept_indices: list[int],
    val_views: set[int],
) -> float | None:
    residuals = []
    for point, original_track_id in zip(optimized_points, kept_indices):
        point2 = point.reshape(1, 3)
        for obs in original_tracks[original_track_id]:
            if obs.image_id not in val_views:
                continue
            xy = project_points(point2, k, rotations[obs.image_id], translations[obs.image_id])[0]
            residuals.extend(xy - obs.xy)
    if not residuals:
        return None
    arr = np.asarray(residuals, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)))


def run_adapter_once(
    k: np.ndarray,
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    points: np.ndarray,
    tracks: list[list[Observation]],
    optimize_views: list[int],
    ba_points: int,
    ba_iters: int,
    backend: str,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, list[int], float, float, str]:
    compact_points, compact, kept_indices, _ = compact_tracks(points, tracks, optimize_views)
    if len(compact_points) == 0:
        return rotations, translations, points[:0], [], float("inf"), float("inf"), "none"

    compact_rot = [rotations[idx].copy() for idx in optimize_views]
    compact_trans = [translations[idx].copy() for idx in optimize_views]
    result = bundle_adjust(
        k,
        compact_rot,
        compact_trans,
        compact_points,
        compact,
        max_points=ba_points,
        max_iterations=ba_iters,
        backend=backend,
    )

    adapted_rot = [rot.copy() for rot in rotations]
    adapted_trans = [trans.copy() for trans in translations]
    for local_id, original_id in enumerate(optimize_views):
        adapted_rot[original_id] = result.rotations[local_id]
        adapted_trans[original_id] = result.translations[local_id]

    return (
        adapted_rot,
        adapted_trans,
        result.points,
        kept_indices,
        result.initial_rmse,
        result.final_rmse,
        result.backend,
    )


def cross_validate_adapter(
    k: np.ndarray,
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    points: np.ndarray,
    tracks: list[list[Observation]],
    train_pool: list[int],
    candidates: list[dict],
    backend: str,
) -> tuple[dict, list[FoldResult]]:
    folds = [train_pool[offset::3] for offset in range(3)]
    results: list[FoldResult] = []
    for candidate in candidates:
        for fold_id, val_views in enumerate(folds):
            opt_views = [idx for idx in train_pool if idx not in set(val_views)]
            print(
                f"[adapter-cv] candidate={candidate['name']} fold={fold_id} "
                f"train={opt_views} val={val_views}",
                flush=True,
            )
            before_val = validation_rmse_for_kept_tracks(
                k,
                rotations,
                translations,
                points,
                tracks,
                list(range(len(points))),
                set(val_views),
            )
            adapted_rot, adapted_trans, adapted_points, kept_indices, train_before, train_after, backend = run_adapter_once(
                k,
                rotations,
                translations,
                points,
                tracks,
                opt_views,
                ba_points=int(candidate["ba_points"]),
                ba_iters=int(candidate["ba_iters"]),
                backend=backend,
            )
            after_val = validation_rmse_for_kept_tracks(
                k,
                adapted_rot,
                adapted_trans,
                adapted_points,
                tracks,
                kept_indices,
                set(val_views),
            )
            results.append(
                FoldResult(
                    candidate=candidate["name"],
                    fold=fold_id,
                    train_views=opt_views,
                    val_views=val_views,
                    train_rmse_before=train_before,
                    train_rmse_after=train_after,
                    val_rmse_before=before_val,
                    val_rmse_after=after_val,
                    optimized_tracks=len(adapted_points),
                    backend=backend,
                )
            )

    def score(candidate: dict) -> float:
        vals = [
            row.val_rmse_after
            for row in results
            if row.candidate == candidate["name"] and row.val_rmse_after is not None and np.isfinite(row.val_rmse_after)
        ]
        return float(np.mean(vals)) if vals else float("inf")

    best = min(candidates, key=score)
    return best, results


def filter_initial_points(
    points: np.ndarray,
    colors: np.ndarray,
    frames: list[Frame],
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    train_pool: list[int],
    k: np.ndarray,
    min_visible_views: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_frames = [frames[idx] for idx in train_pool]
    train_rot = [rotations[idx] for idx in train_pool]
    train_trans = [translations[idx] for idx in train_pool]
    return filter_dense_points_by_cameras(
        points,
        colors,
        train_frames,
        k,
        train_rot,
        train_trans,
        mode="mask",
        min_visible_views=min_visible_views,
    )[:2]


def write_test_split(dataset: Path, test_views: list[int]) -> None:
    names = [f"rgb_{idx:04d}.png" for idx in test_views]
    (dataset / "sparse" / "0" / "test.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def load_existing_colmap(dataset: Path, max_features: int):
    sparse = dataset / "sparse" / "0"
    cameras = read_cameras(sparse / "cameras.txt")
    image_poses = sorted(read_images(sparse / "images.txt"), key=lambda pose: natural_key(Path(pose.name)))
    if not image_poses:
        raise RuntimeError(f"No COLMAP images found in {sparse / 'images.txt'}")
    camera = cameras[image_poses[0].camera_id]
    if camera.model != "PINHOLE":
        raise ValueError(f"Unsupported camera model: {camera.model}")
    fx, fy, cx, cy = camera.params
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    frames: list[Frame] = []
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for pose in image_poses:
        rgba = cv2.imread(str(dataset / "images" / pose.name), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise FileNotFoundError(dataset / "images" / pose.name)
        if rgba.ndim == 2:
            image = cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGR)
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        elif rgba.shape[2] == 4:
            image = rgba[:, :, :3]
            mask = rgba[:, :, 3]
        else:
            image = rgba[:, :, :3]
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        frames.append(Frame(name=pose.name, image=image, mask=(mask > 10).astype(np.uint8) * 255, scale=1.0))
        rotations.append(qvec_to_rotmat(pose.qvec))
        translations.append(pose.tvec.astype(np.float64))

    _, _, xyz, rgb = read_points(sparse / "points3D.txt")
    colors_bgr = rgb[:, ::-1].astype(np.float64)
    print(f"[reuse-colmap] extracting sparse tracks from {dataset}", flush=True)
    rec = reconstruct_with_known_cameras(
        frames,
        k,
        rotations,
        translations,
        max_features=max_features,
        max_reprojection=12.0,
        match_window=1,
        loop_closure=False,
    )
    print(f"[reuse-colmap] sparse tracks={len(rec.tracks)} points={len(rec.points)} rmse={rec.reprojection_rmse:.4f}", flush=True)
    return frames, k, rotations, translations, xyz.astype(np.float64), colors_bgr, rec


def export_dataset(
    name: str,
    frames: list[Frame],
    k: np.ndarray,
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    points: np.ndarray,
    colors: np.ndarray,
    train_pool: list[int],
    test_views: list[int],
    min_visible_views: int,
) -> Path:
    dataset = OUTPUT_ROOT / f"{name}_colmap_eval"
    if dataset.exists():
        return dataset
    filtered_points, filtered_colors = filter_initial_points(
        points,
        colors,
        frames,
        rotations,
        translations,
        train_pool,
        k,
        min_visible_views=min_visible_views,
    )
    write_colmap_text_dataset(
        dataset,
        frames,
        k,
        rotations,
        translations,
        filtered_points,
        filtered_colors,
        alpha_mask_images=True,
    )
    write_test_split(dataset, test_views)
    return dataset


def run_command(command: list[str], log_path: Path | None = None, dry_run: bool = False) -> None:
    print(" ".join(command))
    if dry_run:
        return
    if log_path is None:
        subprocess.run(command, cwd=ROOT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def train_graphdeco(
    dataset: Path,
    model_dir: Path,
    iterations: int,
    conda_env: str,
    dry_run: bool,
) -> Path:
    final_ply = model_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if final_ply.exists():
        return model_dir
    command = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        str(GRAPHDECO_ROOT / "train.py"),
        "-s",
        str(dataset),
        "-m",
        str(model_dir),
        "--eval",
        "--iterations",
        str(iterations),
        "--test_iterations",
        "1000",
        "3000",
        "7000",
        str(iterations),
        "--save_iterations",
        str(iterations),
        "--alpha_bg_weight",
        "2.0",
        "--disable_viewer",
        "--quiet",
    ]
    run_command(command, log_path=model_dir / "train.log", dry_run=dry_run)
    return model_dir


def render_and_evaluate(
    model_dir: Path,
    iterations: int,
    metrics_name: str,
    conda_env: str,
    dry_run: bool,
) -> dict:
    render_dir = model_dir / "test" / f"ours_{iterations}" / "renders"
    gt_dir = model_dir / "test" / f"ours_{iterations}" / "gt"
    metrics_path = OUTPUT_ROOT / f"{metrics_name}_test_render_metrics.json"
    if not render_dir.exists():
        run_command(
            [
                "conda",
                "run",
                "-n",
                conda_env,
                "python",
                str(GRAPHDECO_ROOT / "render.py"),
                "-m",
                str(model_dir),
                "--iteration",
                str(iterations),
                "--skip_train",
                "--quiet",
            ],
            dry_run=dry_run,
        )
    if not metrics_path.exists():
        run_command(
            [
                sys.executable,
                "scripts/evaluate_render_metrics.py",
                "--render-dir",
                str(render_dir),
                "--gt-dir",
                str(gt_dir),
                "--output",
                str(metrics_path),
            ],
            dry_run=dry_run,
        )
    if dry_run:
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def parse_eval_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    rows = []
    pattern = re.compile(r"\[ITER (\d+)\] Evaluating (\w+): L1 ([0-9.eE+-]+) PSNR ([0-9.eE+-]+)")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        rows.append(
            {
                "iteration": int(match.group(1)),
                "split": match.group(2),
                "l1": float(match.group(3)),
                "psnr": float(match.group(4)),
            }
        )
    return rows


def make_contact_sheet(model_dirs: dict[str, Path], iterations: int, dry_run: bool) -> None:
    for name, model_dir in model_dirs.items():
        render_dir = model_dir / "test" / f"ours_{iterations}" / "renders"
        gt_dir = model_dir / "test" / f"ours_{iterations}" / "gt"
        out = PPT_ASSETS / f"vggt_adapter_split_data2_{name}_test_render_sheet.png"
        if out.exists() or dry_run:
            continue
        run_command(
            [
                sys.executable,
                "scripts/make_render_contact_sheet.py",
                "--render-dir",
                str(render_dir),
                "--gt-dir",
                str(gt_dir),
                "--output",
                str(out),
                "--count",
                "4",
                "--thumb-width",
                "220",
            ]
        )


def write_report(summary: dict, fold_rows: list[FoldResult]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fold_csv = OUTPUT_ROOT / "adapter_cross_validation_folds.csv"
    with fold_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate",
                "fold",
                "train_views",
                "val_views",
                "train_rmse_before",
                "train_rmse_after",
                "val_rmse_before",
                "val_rmse_after",
                "optimized_tracks",
                "backend",
            ],
        )
        writer.writeheader()
        for row in fold_rows:
            writer.writerow(
                {
                    "candidate": row.candidate,
                    "fold": row.fold,
                    "train_views": " ".join(map(str, row.train_views)),
                    "val_views": " ".join(map(str, row.val_views)),
                    "train_rmse_before": row.train_rmse_before,
                    "train_rmse_after": row.train_rmse_after,
                    "val_rmse_before": row.val_rmse_before,
                    "val_rmse_after": row.val_rmse_after,
                    "optimized_tracks": row.optimized_tracks,
                    "backend": row.backend,
                }
            )

    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    base = summary["methods"]["baseline"]
    adapter = summary["methods"]["geometry_adapter"]
    base_render = base.get("render_metrics", {})
    adapter_render = adapter.get("render_metrics", {})
    lines = [
        "# Frozen VGGT + Geometry Adapter Held-out Experiment",
        "",
        "Data2 uses a fixed 8/4/4 protocol: the last-level test views are never used for adapter fitting or 3DGS training.",
        "",
        f"- Test views: {summary['splits']['test_views']}",
        f"- Train/validation pool: {summary['splits']['train_pool']}",
        f"- Adapter selected by 3-fold 8/4 validation: `{summary['adapter_selection']['best_candidate']}`",
        "",
        "## Geometry Loss",
        "",
        f"- Baseline VGGT train-pool reprojection RMSE: {base['train_pool_rmse']:.4f} px",
        f"- Adapter refit train-pool RMSE before/after: {adapter['refit_initial_rmse']:.4f} -> {adapter['refit_final_rmse']:.4f} px",
        f"- Mean validation RMSE after adapter selection: {summary['adapter_selection']['best_mean_val_rmse']:.4f} px",
        "",
        "## Held-out Test Render Metrics",
        "",
        "| Method | Test PSNR | Test SSIM | Test FG MAE | Test images |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Frozen VGGT baseline | {base_render.get('mean_psnr')} | {base_render.get('mean_ssim')} | {base_render.get('mean_fg_mae')} | {base_render.get('count')} |",
        f"| Frozen VGGT + geometry adapter | {adapter_render.get('mean_psnr')} | {adapter_render.get('mean_ssim')} | {adapter_render.get('mean_fg_mae')} | {adapter_render.get('count')} |",
        "",
        "## Files",
        "",
        f"- Fold losses: `{fold_csv}`",
        f"- Baseline model: `{base['model_dir']}`",
        f"- Adapter model: `{adapter['model_dir']}`",
    ]
    (OUTPUT_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global OUTPUT_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DATA2)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--vggt-max-points", type=int, default=50000)
    parser.add_argument("--point-min-visible-views", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=2500)
    parser.add_argument("--adapter-backend", choices=["legacy", "scipy"], default="legacy")
    parser.add_argument("--graphdeco-env", default="vggt")
    parser.add_argument(
        "--reuse-colmap",
        type=Path,
        default=DEFAULT_REUSE_COLMAP,
        help="Reuse an existing VGGT COLMAP export for frozen cameras and dense points. Pass an empty path to run VGGT again.",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    OUTPUT_ROOT = args.output_root
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(PPT_ASSETS)

    test_views = [3, 7, 11, 15]
    train_pool = [idx for idx in range(16) if idx not in set(test_views)]
    candidates = [
        {"name": "adapter_ba40_i3", "ba_points": 40, "ba_iters": 3},
        {"name": "adapter_ba80_i4", "ba_points": 80, "ba_iters": 4},
    ]

    reuse_colmap = args.reuse_colmap if str(args.reuse_colmap) else None
    if reuse_colmap is not None and reuse_colmap.exists():
        frames, k, rotations, translations, raw_points, raw_colors, rec = load_existing_colmap(
            reuse_colmap,
            max_features=args.max_features,
        )
        frozen_source = f"existing_colmap:{reuse_colmap}"
    else:
        vggt_dir = OUTPUT_ROOT / "frozen_vggt_p0"
        vggt = run_vggt(
            args.source,
            vggt_dir,
            model_path=ROOT / ".models" / "VGGT-1B",
            max_images=0,
            max_size=0,
            max_features=0,
            max_points=args.vggt_max_points,
            confidence_percentile=0.0,
            mode="pad",
            input_background_mode="none",
            input_mask_erode=0,
            allow_cpu=False,
            skip_sparse=False,
            sparse_match_window=1,
            sparse_loop_closure=False,
        )
        rec = vggt.reconstruction
        frames = vggt.frames
        k = rec.camera_matrix
        rotations = rec.rotations
        translations = rec.translations
        raw_points = vggt.raw_points
        raw_colors = vggt.raw_colors
        frozen_source = "fresh_vggt_p0"

    best, folds = cross_validate_adapter(
        k,
        rotations,
        translations,
        rec.points,
        rec.tracks,
        train_pool,
        candidates,
        backend=args.adapter_backend,
    )
    best_vals = [
        row.val_rmse_after
        for row in folds
        if row.candidate == best["name"] and row.val_rmse_after is not None and np.isfinite(row.val_rmse_after)
    ]

    adapted_rot, adapted_trans, adapted_points, kept_indices, refit_initial, refit_final, adapter_backend = run_adapter_once(
        k,
        rotations,
        translations,
        rec.points,
        rec.tracks,
        train_pool,
        ba_points=int(best["ba_points"]),
        ba_iters=int(best["ba_iters"]),
        backend=args.adapter_backend,
    )

    baseline_dataset = export_dataset(
        "baseline",
        frames,
        k,
        rotations,
        translations,
        raw_points,
        raw_colors,
        train_pool,
        test_views,
        min_visible_views=args.point_min_visible_views,
    )
    adapter_dataset = export_dataset(
        "geometry_adapter",
        frames,
        k,
        adapted_rot,
        adapted_trans,
        raw_points,
        raw_colors,
        train_pool,
        test_views,
        min_visible_views=args.point_min_visible_views,
    )

    baseline_model = OUTPUT_ROOT / f"baseline_graphdeco_{args.iterations}"
    adapter_model = OUTPUT_ROOT / f"geometry_adapter_graphdeco_{args.iterations}"
    if not args.skip_train:
        train_graphdeco(baseline_dataset, baseline_model, args.iterations, args.graphdeco_env, args.dry_run)
        train_graphdeco(adapter_dataset, adapter_model, args.iterations, args.graphdeco_env, args.dry_run)
        baseline_render = render_and_evaluate(
            baseline_model,
            args.iterations,
            "baseline",
            args.graphdeco_env,
            args.dry_run,
        )
        adapter_render = render_and_evaluate(
            adapter_model,
            args.iterations,
            "geometry_adapter",
            args.graphdeco_env,
            args.dry_run,
        )
        make_contact_sheet({"baseline": baseline_model, "geometry_adapter": adapter_model}, args.iterations, args.dry_run)
    else:
        baseline_render = {}
        adapter_render = {}

    summary = {
        "source": str(args.source),
        "frozen_vggt_source": frozen_source,
        "iterations": int(args.iterations),
        "splits": {
            "protocol": "8/4/4",
            "test_views": test_views,
            "train_pool": train_pool,
            "folds": [{"fold": i, "val_views": train_pool[i::3]} for i in range(3)],
        },
        "adapter_selection": {
            "candidates": candidates,
            "best_candidate": best["name"],
            "best_mean_val_rmse": float(np.mean(best_vals)) if best_vals else None,
            "fold_csv": str(OUTPUT_ROOT / "adapter_cross_validation_folds.csv"),
        },
        "methods": {
            "baseline": {
                "dataset": str(baseline_dataset),
                "model_dir": str(baseline_model),
                "train_pool_rmse": reprojection_rmse(
                    k,
                    rotations,
                    translations,
                    rec.points,
                    rec.tracks,
                    set(train_pool),
                ),
                "test_track_rmse": reprojection_rmse(
                    k,
                    rotations,
                    translations,
                    rec.points,
                    rec.tracks,
                    set(test_views),
                ),
                "graphdeco_eval_log": parse_eval_log(baseline_model / "train.log"),
                "render_metrics": baseline_render,
            },
            "geometry_adapter": {
                "dataset": str(adapter_dataset),
                "model_dir": str(adapter_model),
                "adapter_backend": adapter_backend,
                "refit_initial_rmse": refit_initial,
                "refit_final_rmse": refit_final,
                "optimized_tracks": len(adapted_points),
                "optimized_track_indices": len(kept_indices),
                "test_track_rmse": reprojection_rmse(
                    k,
                    adapted_rot,
                    adapted_trans,
                    rec.points,
                    rec.tracks,
                    set(test_views),
                ),
                "graphdeco_eval_log": parse_eval_log(adapter_model / "train.log"),
                "render_metrics": adapter_render,
            },
        },
    }
    write_report(summary, folds)
    print(OUTPUT_ROOT / "summary.md")


if __name__ == "__main__":
    main()
