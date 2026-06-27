from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from reconstruct3d.ba import bundle_adjust
from reconstruct3d.io_utils import Frame, ensure_dir, load_image_sequence, write_colmap_text_dataset
from reconstruct3d.sfm import reconstruct_with_known_cameras
from run_project import filter_dense_points_by_cameras
from scripts.filter_colmap_points_by_alpha import qvec_to_rotmat, read_cameras, read_images


DATA1 = ROOT / "大作业数据" / "数据1-人体"
DATA2 = ROOT / "大作业数据" / "数据2-人体"
DATA1_TARGET = ROOT / "outputs" / "official_data1_colmap_50k_p0_masked_strict_ba_fgstrict"
GRAPHDECO_ROOT = Path("/tmp/gaussian-splatting-official")
OUT = ROOT / "outputs" / "vggt_data1_finetune_data2_transfer"
PPT = ROOT / "ppt_assets"


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def image_paths_from_source(source: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted([p for p in source.iterdir() if p.suffix.lower() in exts and not p.name.startswith("msk_")], key=natural_key)


def colmap_pose_target(dataset: Path, device: str):
    from vggt.utils.pose_enc import extri_intri_to_pose_encoding

    sparse = dataset / "sparse" / "0"
    cameras = read_cameras(sparse / "cameras.txt")
    poses = sorted(read_images(sparse / "images.txt"), key=lambda pose: natural_key(Path(pose.name)))
    camera = cameras[poses[0].camera_id]
    fx, fy, cx, cy = camera.params
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    extrinsics = []
    intrinsics = []
    for pose in poses:
        ext = np.concatenate([qvec_to_rotmat(pose.qvec), pose.tvec.reshape(3, 1)], axis=1).astype(np.float32)
        extrinsics.append(ext)
        intrinsics.append(k)
    extri_t = torch.from_numpy(np.stack(extrinsics, axis=0)).unsqueeze(0).to(device)
    intri_t = torch.from_numpy(np.stack(intrinsics, axis=0)).unsqueeze(0).to(device)
    pose_enc = extri_intri_to_pose_encoding(
        extri_t,
        intri_t,
        image_size_hw=(camera.height, camera.width),
    )
    return pose_enc, extri_t, intri_t, (camera.height, camera.width)


def trainable_camera_adapter_params(model, mode: str) -> list[str]:
    if mode == "pose_tail":
        prefixes = (
            "camera_head.pose_branch",
            "camera_head.poseLN_modulation",
            "camera_head.embed_pose",
            "camera_head.token_norm",
            "camera_head.trunk_norm",
        )
    elif mode == "pose_branch":
        prefixes = ("camera_head.pose_branch",)
    else:
        raise ValueError(mode)
    return [name for name, _ in model.named_parameters() if name.startswith(prefixes)]


def set_trainable(model, trainable_names: set[str]) -> None:
    for name, param in model.named_parameters():
        param.requires_grad_(name in trainable_names)


def load_preprocessed_images(paths: list[Path], mode: str, device: str) -> torch.Tensor:
    from vggt.utils.load_fn import load_and_preprocess_images

    return load_and_preprocess_images([str(path) for path in paths], mode=mode).to(device)


def finetune_on_data1(args: argparse.Namespace) -> Path:
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("This fine-tune experiment requires CUDA for VGGT-1B.")
    torch.manual_seed(args.seed)

    run_dir = ensure_dir(args.output / "data1_finetune")
    checkpoint = run_dir / "camera_adapter.pt"
    if checkpoint.exists() and not args.force_finetune:
        return checkpoint

    model = VGGT.from_pretrained(str(args.vggt_model)).to(device)
    model.train()
    target_pose, target_extri, target_intri, image_size_hw = colmap_pose_target(args.data1_target, device)
    image_paths = image_paths_from_source(args.data1_target / "images")
    images = load_preprocessed_images(image_paths, args.preprocess_mode, device)

    with torch.no_grad():
        aggregated_tokens_list, _ = model.aggregator(images.unsqueeze(0) if images.ndim == 4 else images)
        cached_tokens = aggregated_tokens_list[-1].detach()
        baseline_pose = model.camera_head([cached_tokens])[-1].detach()

    trainable_names = set(trainable_camera_adapter_params(model, args.adapter_mode))
    set_trainable(model, trainable_names)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    rows = []
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        pred_pose = model.camera_head([cached_tokens])[-1]
        pred_extri, pred_intri = pose_encoding_to_extri_intri(pred_pose, image_size_hw=image_size_hw, build_intrinsics=True)
        loss_t = torch.nn.functional.smooth_l1_loss(pred_extri[..., :3, 3], target_extri[..., :3, 3])
        loss_r = torch.nn.functional.smooth_l1_loss(pred_extri[..., :3, :3], target_extri[..., :3, :3])
        pred_f = torch.stack([pred_intri[..., 0, 0], pred_intri[..., 1, 1]], dim=-1).log()
        target_f = torch.stack([target_intri[..., 0, 0], target_intri[..., 1, 1]], dim=-1).log()
        loss_f = torch.nn.functional.smooth_l1_loss(pred_f, target_f)
        loss_anchor = torch.nn.functional.smooth_l1_loss(pred_pose, baseline_pose)
        loss = args.translation_weight * loss_t + args.rotation_weight * loss_r + args.focal_weight * loss_f + args.anchor_weight * loss_anchor
        if step > 0:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        if step % args.log_every == 0 or step == args.steps:
            row = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "loss_t": float(loss_t.detach().cpu()),
                "loss_r": float(loss_r.detach().cpu()),
                "loss_f": float(loss_f.detach().cpu()),
                "loss_anchor": float(loss_anchor.detach().cpu()),
            }
            rows.append(row)
            print(f"[finetune] {row}", flush=True)

    state = {
        "adapter_mode": args.adapter_mode,
        "trainable_names": sorted(trainable_names),
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if any(name.startswith(prefix) for prefix in [n.rsplit(".", 1)[0] for n in trainable_names])},
        "args": vars(args),
    }
    torch.save(state, checkpoint)
    (run_dir / "loss.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return checkpoint


def load_adapter_weights(model, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    print(f"[adapter-load] loaded={len(payload['state_dict'])} missing={len(missing)} unexpected={len(unexpected)}", flush=True)


def mask_stack_for_frames(frames: list[Frame], height: int, width: int) -> np.ndarray:
    masks = []
    for frame in frames:
        if frame.mask is None:
            mask = np.ones(frame.image.shape[:2], dtype=np.uint8) * 255
        else:
            mask = frame.mask
        masks.append(cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 10)
    return np.stack(masks, axis=0)


def processed_frames_from_prediction(input_images: torch.Tensor, masks: np.ndarray) -> list[Frame]:
    images_rgb = input_images.detach().cpu().numpy()
    frames = []
    for index, image_rgb in enumerate(np.transpose(images_rgb, (0, 2, 3, 1))):
        image_bgr = np.clip(image_rgb[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
        mask_u8 = masks[index].astype(np.uint8) * 255
        frames.append(Frame(name=f"rgb_{index:04d}.png", image=image_bgr, mask=mask_u8, scale=1.0))
    return frames


def spatial_sample(points: np.ndarray, colors: np.ndarray, conf: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(points).all(axis=1) & np.isfinite(conf)
    points = points[finite]
    colors = colors[finite]
    conf = conf[finite]
    if max_points <= 0 or len(points) <= max_points:
        return points.astype(np.float64), colors.astype(np.float64)
    order = np.argsort(conf)[::-1]
    chosen = order[np.linspace(0, len(order) - 1, max_points, dtype=np.int64)]
    return points[chosen].astype(np.float64), colors[chosen].astype(np.float64)


def run_model_export_data2(args: argparse.Namespace, checkpoint: Path | None, name: str) -> dict:
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ensure_dir(args.output / name)
    dataset = args.output / f"{name}_colmap_eval"
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and dataset.exists() and not args.force_eval:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    model = VGGT.from_pretrained(str(args.vggt_model)).to(device)
    if checkpoint is not None:
        load_adapter_weights(model, checkpoint)
    model.eval()

    source_frames = load_image_sequence(args.data2, max_images=0, max_size=0, use_masks=True)
    image_paths = image_paths_from_source(args.data2)
    images = load_preprocessed_images(image_paths, args.preprocess_mode, device)
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            pred = model(images)
    h, w = int(images.shape[-2]), int(images.shape[-1])
    extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], image_size_hw=(h, w), build_intrinsics=True)
    extri_np = extri[0].detach().cpu().numpy()
    intri_np = intri[0].detach().cpu().numpy()
    rotations = [ext[:3, :3].astype(np.float64) for ext in extri_np]
    translations = [ext[:3, 3].astype(np.float64) for ext in extri_np]
    k = intri_np[0].astype(np.float64)

    world_points = pred["world_points"][0].detach().cpu().numpy()
    world_conf = pred["world_points_conf"][0].detach().cpu().numpy()
    input_images = pred["images"][0].detach().cpu().numpy()
    colors = (np.transpose(input_images, (0, 2, 3, 1)).reshape(-1, 3) * 255.0)[:, ::-1]
    flat_points = world_points.reshape(-1, 3)
    flat_conf = world_conf.reshape(-1)
    masks = mask_stack_for_frames(source_frames, world_points.shape[1], world_points.shape[2])
    frames = processed_frames_from_prediction(pred["images"][0], masks)
    mask_stack = masks.reshape(-1)
    flat_points = flat_points[mask_stack]
    colors = colors[mask_stack]
    flat_conf = flat_conf[mask_stack]
    raw_points, raw_colors = spatial_sample(flat_points, colors, flat_conf, args.vggt_max_points)

    rec = reconstruct_with_known_cameras(
        frames,
        k,
        rotations,
        translations,
        max_features=args.max_features,
        max_reprojection=12.0,
        match_window=1,
        loop_closure=False,
    )
    if args.ba_backend != "none" and len(rec.points) > 0:
        ba = bundle_adjust(k, rotations, translations, rec.points, rec.tracks, max_points=args.ba_points, max_iterations=args.ba_iters, backend=args.ba_backend)
        rotations_out = ba.rotations
        translations_out = ba.translations
        ba_initial = ba.initial_rmse
        ba_final = ba.final_rmse
        sparse_points = len(ba.points)
    else:
        ba = None
        rotations_out = rotations
        translations_out = translations
        ba_initial = None
        ba_final = None
        sparse_points = len(rec.points)

    test_views = [3, 7, 11, 15]
    train_views = [i for i in range(len(frames)) if i not in set(test_views)]
    train_points, train_colors, _ = filter_dense_points_by_cameras(
        raw_points,
        raw_colors,
        [frames[i] for i in train_views],
        k,
        [rotations_out[i] for i in train_views],
        [translations_out[i] for i in train_views],
        mode="mask",
        min_visible_views=args.point_min_visible_views,
    )
    if dataset.exists() and args.force_eval:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {dataset}")
    if not dataset.exists():
        write_colmap_text_dataset(dataset, frames, k, rotations_out, translations_out, train_points, train_colors, alpha_mask_images=True)
        (dataset / "sparse" / "0" / "test.txt").write_text("\n".join(f"rgb_{i:04d}.png" for i in test_views) + "\n", encoding="utf-8")

    metrics = {
        "name": name,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "reprojection_rmse_before_ba": float(rec.reprojection_rmse) if np.isfinite(rec.reprojection_rmse) else None,
        "ba_initial_rmse": ba_initial,
        "ba_final_rmse": ba_final,
        "sparse_points": sparse_points,
        "pair_inliers_sum": int(sum(rec.pair_inliers)),
        "dense_points": int(len(raw_points)),
        "export_points": int(len(train_points)),
        "dataset": str(dataset),
        "output": str(out_dir),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def run_command(command: list[str], cwd: Path = ROOT, log_path: Path | None = None) -> None:
    print(" ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, cwd=cwd, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def graphdeco_eval(args: argparse.Namespace, metrics: dict) -> dict:
    model_dir = args.output / f"{metrics['name']}_graphdeco_{args.graphdeco_iterations}"
    render_metrics_path = args.output / f"{metrics['name']}_test_render_metrics.json"
    final_ply = model_dir / "point_cloud" / f"iteration_{args.graphdeco_iterations}" / "point_cloud.ply"
    if not final_ply.exists():
        run_command(
            [
                "conda",
                "run",
                "-n",
                args.graphdeco_env,
                "python",
                str(GRAPHDECO_ROOT / "train.py"),
                "-s",
                metrics["dataset"],
                "-m",
                str(model_dir),
                "--eval",
                "--iterations",
                str(args.graphdeco_iterations),
                "--test_iterations",
                "1000",
                "3000",
                str(args.graphdeco_iterations),
                "--save_iterations",
                str(args.graphdeco_iterations),
                "--alpha_bg_weight",
                "2.0",
                "--disable_viewer",
                "--quiet",
            ],
            log_path=model_dir / "train.log",
        )
    render_dir = model_dir / "test" / f"ours_{args.graphdeco_iterations}" / "renders"
    gt_dir = model_dir / "test" / f"ours_{args.graphdeco_iterations}" / "gt"
    if not render_dir.exists():
        run_command(
            [
                "conda",
                "run",
                "-n",
                args.graphdeco_env,
                "python",
                str(GRAPHDECO_ROOT / "render.py"),
                "-m",
                str(model_dir),
                "--iteration",
                str(args.graphdeco_iterations),
                "--skip_train",
                "--quiet",
            ]
        )
    if not render_metrics_path.exists():
        run_command(
            [
                sys.executable,
                "scripts/evaluate_render_metrics.py",
                "--render-dir",
                str(render_dir),
                "--gt-dir",
                str(gt_dir),
                "--output",
                str(render_metrics_path),
            ]
        )
    metrics["graphdeco_model"] = str(model_dir)
    metrics["render_metrics"] = json.loads(render_metrics_path.read_text(encoding="utf-8"))
    return metrics


def write_summary(args: argparse.Namespace, baseline: dict, finetuned: dict, checkpoint: Path) -> None:
    summary = {
        "experiment": "data1 camera-head adapter fine-tune, data2 transfer",
        "checkpoint": str(checkpoint),
        "baseline": baseline,
        "finetuned": finetuned,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    b_render = baseline.get("render_metrics", {})
    f_render = finetuned.get("render_metrics", {})
    lines = [
        "# Data1 Fine-tuned VGGT Adapter -> Data2 Transfer",
        "",
        f"- Adapter checkpoint: `{checkpoint}`",
        f"- Adapter mode: `{args.adapter_mode}`",
        f"- Trainable VGGT modules are camera-head tail modules, not the full 1B backbone.",
        "",
        "## Data2 Geometry",
        "",
        "| Method | BA final RMSE | Sparse points | Pair inliers | Export points |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Frozen VGGT baseline | {baseline.get('ba_final_rmse')} | {baseline.get('sparse_points')} | {baseline.get('pair_inliers_sum')} | {baseline.get('export_points')} |",
        f"| Data1-finetuned adapter | {finetuned.get('ba_final_rmse')} | {finetuned.get('sparse_points')} | {finetuned.get('pair_inliers_sum')} | {finetuned.get('export_points')} |",
        "",
        "## Data2 Held-out Render",
        "",
        "| Method | Test PSNR | Test SSIM | Test FG MAE |",
        "| --- | ---: | ---: | ---: |",
        f"| Frozen VGGT baseline | {b_render.get('mean_psnr')} | {b_render.get('mean_ssim')} | {b_render.get('mean_fg_mae')} |",
        f"| Data1-finetuned adapter | {f_render.get('mean_psnr')} | {f_render.get('mean_ssim')} | {f_render.get('mean_fg_mae')} |",
    ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data1", type=Path, default=DATA1)
    parser.add_argument("--data2", type=Path, default=DATA2)
    parser.add_argument("--data1-target", type=Path, default=DATA1_TARGET)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--vggt-model", type=Path, default=ROOT / ".models" / "VGGT-1B")
    parser.add_argument("--preprocess-mode", choices=["pad", "crop"], default="pad")
    parser.add_argument("--adapter-mode", choices=["pose_tail", "pose_branch"], default="pose_tail")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--translation-weight", type=float, default=1.0)
    parser.add_argument("--rotation-weight", type=float, default=1.0)
    parser.add_argument("--focal-weight", type=float, default=0.05)
    parser.add_argument("--anchor-weight", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--vggt-max-points", type=int, default=50000)
    parser.add_argument("--max-features", type=int, default=2500)
    parser.add_argument("--ba-points", type=int, default=120)
    parser.add_argument("--ba-iters", type=int, default=0)
    parser.add_argument("--ba-backend", choices=["scipy", "legacy", "none"], default="scipy")
    parser.add_argument("--point-min-visible-views", type=int, default=4)
    parser.add_argument("--run-graphdeco", action="store_true")
    parser.add_argument("--graphdeco-env", default="vggt")
    parser.add_argument("--graphdeco-iterations", type=int, default=7000)
    parser.add_argument("--force-finetune", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output)
    checkpoint = finetune_on_data1(args)
    baseline = run_model_export_data2(args, checkpoint=None, name="data2_frozen_vggt")
    finetuned = run_model_export_data2(args, checkpoint=checkpoint, name="data2_data1_finetuned")
    if args.run_graphdeco:
        baseline = graphdeco_eval(args, baseline)
        finetuned = graphdeco_eval(args, finetuned)
    write_summary(args, baseline, finetuned, checkpoint)
    print(args.output / "summary.md")


if __name__ == "__main__":
    main()
