from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "vggt_adapter_split_data2"
PPT = ROOT / "ppt_assets"


def main() -> None:
    PPT.mkdir(exist_ok=True)
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    folds = list(csv.DictReader((OUT / "adapter_cross_validation_folds.csv").open(encoding="utf-8")))
    best = summary["adapter_selection"]["best_candidate"]
    best_folds = [row for row in folds if row["candidate"] == best]

    labels = [f"fold {row['fold']}" for row in best_folds]
    before = [float(row["val_rmse_before"]) for row in best_folds]
    after = [float(row["val_rmse_after"]) for row in best_folds]
    x = list(range(len(labels)))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars1 = ax.bar([i - width / 2 for i in x], before, width, label="before adapter", color="#9ca3af")
    bars2 = ax.bar([i + width / 2 for i in x], after, width, label="after adapter", color="#2563eb")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Validation reprojection RMSE (px)")
    ax.set_title("Data2 frozen VGGT geometry adapter: 8/4 validation")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(before + after) * 1.18)
    for bars in (bars1, bars2):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    cv_path = PPT / "vggt_adapter_split_data2_cv_rmse.png"
    fig.savefig(cv_path, dpi=220)
    plt.close(fig)

    methods = ["baseline", "geometry_adapter"]
    names = ["frozen VGGT", "VGGT + adapter"]
    colors = ["#6b7280", "#2563eb"]
    psnr = [summary["methods"][method]["render_metrics"]["mean_psnr"] for method in methods]
    ssim = [summary["methods"][method]["render_metrics"]["mean_ssim"] for method in methods]
    fg_mae = [summary["methods"][method]["render_metrics"]["mean_fg_mae"] for method in methods]

    fig, ax1 = plt.subplots(figsize=(7.4, 4.6))
    bars = ax1.bar(names, psnr, color=colors, width=0.55)
    ax1.set_ylabel("Held-out test PSNR (dB)")
    ax1.set_ylim(0, max(psnr) * 1.2)
    for bar, value in zip(bars, psnr):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.45, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    ax2 = ax1.twinx()
    ax2.plot(names, fg_mae, color="#111827", marker="o", linewidth=2, label="FG MAE")
    ax2.set_ylabel("Foreground MAE")
    ax2.set_ylim(0, max(fg_mae) * 1.55)
    for i, value in enumerate(fg_mae):
        ax2.text(i, value + 0.00025, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    ax1.set_title("Data2 held-out test render metrics after 3DGS 7k")
    fig.tight_layout()
    render_path = PPT / "vggt_adapter_split_data2_render_metrics.png"
    fig.savefig(render_path, dpi=220)
    plt.close(fig)

    print(cv_path)
    print(render_path)


if __name__ == "__main__":
    main()
