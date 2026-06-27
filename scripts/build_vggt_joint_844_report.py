from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "vggt_adapter_split_data2"
PPT = ROOT / "ppt_assets"


def count_ply_vertices(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        for line in handle:
            text = line.decode("ascii", errors="ignore").strip()
            if text.startswith("element vertex "):
                return int(text.split()[-1])
            if text == "end_header":
                return None
    return None


def rounded(value: object, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def main() -> None:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    joint_metrics = json.loads((OUT / "joint_pose_test_render_metrics.json").read_text(encoding="utf-8"))
    joint_pose = torch.load(OUT / "joint_pose_graphdeco_7000" / "joint_pose_adapter_7000.pt", map_location="cpu")

    rows = [
        {
            "method": "Frozen VGGT baseline",
            "type": "No adapter",
            "geometry_rmse": summary["methods"]["baseline"]["train_pool_rmse"],
            "test_psnr": summary["methods"]["baseline"]["render_metrics"]["mean_psnr"],
            "test_ssim": summary["methods"]["baseline"]["render_metrics"]["mean_ssim"],
            "fg_mae": summary["methods"]["baseline"]["render_metrics"]["mean_fg_mae"],
            "gaussians": count_ply_vertices(OUT / "baseline_graphdeco_7000" / "point_cloud" / "iteration_7000" / "point_cloud.ply"),
        },
        {
            "method": "Frozen VGGT + geometry adapter",
            "type": "Sparse geometry loss",
            "geometry_rmse": summary["methods"]["geometry_adapter"]["refit_final_rmse"],
            "test_psnr": summary["methods"]["geometry_adapter"]["render_metrics"]["mean_psnr"],
            "test_ssim": summary["methods"]["geometry_adapter"]["render_metrics"]["mean_ssim"],
            "fg_mae": summary["methods"]["geometry_adapter"]["render_metrics"]["mean_fg_mae"],
            "gaussians": count_ply_vertices(OUT / "geometry_adapter_graphdeco_7000" / "point_cloud" / "iteration_7000" / "point_cloud.ply"),
        },
        {
            "method": "Frozen VGGT + joint 3DGS pose adapter",
            "type": "3DGS photometric loss",
            "geometry_rmse": None,
            "test_psnr": joint_metrics["mean_psnr"],
            "test_ssim": joint_metrics["mean_ssim"],
            "fg_mae": joint_metrics["mean_fg_mae"],
            "gaussians": count_ply_vertices(OUT / "joint_pose_graphdeco_7000" / "point_cloud" / "iteration_7000" / "point_cloud.ply"),
            "rot_abs_max": float(joint_pose["rot"].abs().max()),
            "trans_abs_max": float(joint_pose["trans"].abs().max()),
            "rot_mean_norm": float(joint_pose["rot"].norm(dim=1).mean()),
            "trans_mean_norm": float(joint_pose["trans"].norm(dim=1).mean()),
        },
    ]

    report = {
        "protocol": summary["splits"],
        "rows": rows,
        "joint_pose_note": "Joint pose adapter is optimized through 3DGS photometric loss by applying a differentiable per-view pose transform to Gaussian means during training. Test views are held out.",
    }
    (OUT / "joint_844_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# VGGT / 3DGS Joint Optimization on Data2 8/4/4",
        "",
        "- Test views: `[3, 7, 11, 15]`",
        "- Train/validation pool: `[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]`",
        "- Joint method: freeze VGGT, optimize per-train-view pose adapter with 3DGS photometric loss.",
        "",
        "| Method | Optimization target | Geometry RMSE | Test PSNR | Test SSIM | FG MAE | Gaussians |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['type']} | {rounded(row.get('geometry_rmse'), 3)} | "
            f"{rounded(row['test_psnr'], 3)} | {rounded(row['test_ssim'], 4)} | {rounded(row['fg_mae'], 5)} | {row['gaussians']} |"
        )
    lines.extend(
        [
            "",
            "## Joint Pose Delta",
            "",
            f"- max rotation delta: `{rows[-1]['rot_abs_max']:.6f}` rad",
            f"- mean rotation norm: `{rows[-1]['rot_mean_norm']:.6f}` rad",
            f"- max translation delta: `{rows[-1]['trans_abs_max']:.6f}`",
            f"- mean translation norm: `{rows[-1]['trans_mean_norm']:.6f}`",
            "",
            "## Conclusion",
            "",
            "The joint 3DGS pose adapter is actively optimized, but it lowers held-out render quality versus the frozen VGGT baseline. In this small 16-view data2 setting, render-loss-driven pose adaptation overfits the training views and does not improve generalization to the fixed held-out views.",
        ]
    )
    (OUT / "joint_844_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    PPT.mkdir(exist_ok=True)
    labels = ["baseline", "geometry", "joint"]
    psnr = [row["test_psnr"] for row in rows]
    ssim = [row["test_ssim"] for row in rows]
    colors = ["#6b7280", "#2563eb", "#dc2626"]
    fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
    bars = ax1.bar(labels, psnr, color=colors, width=0.58)
    ax1.set_ylabel("Held-out test PSNR (dB)")
    ax1.set_ylim(0, max(psnr) * 1.22)
    for bar, value in zip(bars, psnr):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.45, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    ax2 = ax1.twinx()
    ax2.plot(labels, ssim, color="#111827", marker="o", linewidth=2)
    ax2.set_ylabel("SSIM")
    ax2.set_ylim(min(ssim) - 0.01, 1.0)
    for index, value in enumerate(ssim):
        ax2.text(index, value + 0.001, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    ax1.set_title("Data2 8/4/4: VGGT adapters vs held-out render")
    fig.tight_layout()
    chart = PPT / "vggt_joint_844_render_metrics.png"
    fig.savefig(chart, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    gaussians = [row["gaussians"] for row in rows]
    bars = ax.bar(labels, gaussians, color=colors, width=0.58)
    ax.set_ylabel("Gaussians after 7k")
    ax.set_title("Data2 8/4/4: model size after 3DGS training")
    ax.set_ylim(0, max(gaussians) * 1.18)
    for bar, value in zip(bars, gaussians):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1400, str(value), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    size_chart = PPT / "vggt_joint_844_gaussian_count.png"
    fig.savefig(size_chart, dpi=220)
    plt.close(fig)

    print(OUT / "joint_844_report.md")
    print(chart)
    print(size_chart)


if __name__ == "__main__":
    main()
