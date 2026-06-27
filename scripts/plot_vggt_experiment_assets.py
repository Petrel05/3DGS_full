from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    csv_path = Path("outputs/vggt_data2_network_experiments/vggt_data2_experiment_report.csv")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    labels = ["baseline", "p0", "gray bg", "fg crop", "wide tracks"]
    colors = ["#6b7280", "#9ca3af", "#2f80ed", "#27ae60", "#f2994a"]
    rmse = [float(row["ba_final_rmse"]) for row in rows]
    points = [int(row["sparse_points"]) for row in rows]

    out = Path("ppt_assets")
    out.mkdir(exist_ok=True)

    plt.figure(figsize=(9, 4.8))
    bars = plt.bar(labels, rmse, color=colors)
    plt.ylabel("BA final reprojection RMSE (px)")
    plt.title("Data2 VGGT-side Optimization: Geometry Metric")
    plt.ylim(0, max(rmse) * 1.18)
    for bar, value in zip(bars, rmse):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(out / "vggt_data2_ba_rmse_experiments.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    bars = plt.bar(labels, points, color=colors)
    plt.ylabel("Sparse BA points")
    plt.title("Data2 VGGT-side Optimization: Track Geometry Support")
    plt.ylim(0, max(points) * 1.18)
    for bar, value in zip(bars, points):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 8, str(value), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(out / "vggt_data2_sparse_points_experiments.png", dpi=220)
    plt.close()

    render_rows = [(label, row) for label, row in zip(labels, rows) if row["render_psnr"]]
    psnr_labels = [label for label, _ in render_rows]
    psnr = [float(row["render_psnr"]) for _, row in render_rows]
    ssim = [float(row["render_ssim"]) for _, row in render_rows]
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    x = list(range(len(psnr_labels)))
    bars = ax1.bar(x, psnr, color=["#6b7280", "#2f80ed", "#27ae60"], width=0.55)
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_xticks(x, psnr_labels)
    ax1.set_ylim(0, max(psnr) * 1.18)
    for bar, value in zip(bars, psnr):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}", ha="center", va="bottom", fontsize=10)
    ax2 = ax1.twinx()
    ax2.plot(x, ssim, color="#111827", marker="o", linewidth=2)
    ax2.set_ylabel("SSIM")
    ax2.set_ylim(min(ssim) - 0.01, 1.0)
    for i, value in enumerate(ssim):
        ax2.text(i, value + 0.001, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    plt.title("Data2 Offline Render/GT Metrics after Graphdeco 30k")
    fig.tight_layout()
    fig.savefig(out / "vggt_data2_render_metrics_experiments.png", dpi=220)
    plt.close(fig)

    print(out / "vggt_data2_ba_rmse_experiments.png")
    print(out / "vggt_data2_sparse_points_experiments.png")
    print(out / "vggt_data2_render_metrics_experiments.png")


if __name__ == "__main__":
    main()
