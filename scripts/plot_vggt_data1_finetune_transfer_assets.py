from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("tail 80", ROOT / "outputs" / "vggt_data1_finetune_data2_transfer"),
    ("tail 30 + anchor", ROOT / "outputs" / "vggt_data1_finetune_data2_transfer_conservative"),
    ("branch 80", ROOT / "outputs" / "vggt_data1_finetune_data2_transfer_posebranch"),
]
PPT = ROOT / "ppt_assets"


def load_metrics(root: Path):
    base = json.loads((root / "data2_frozen_vggt" / "metrics.json").read_text(encoding="utf-8"))
    tuned = json.loads((root / "data2_data1_finetuned" / "metrics.json").read_text(encoding="utf-8"))
    return base, tuned


def main() -> None:
    PPT.mkdir(exist_ok=True)
    labels = []
    tuned_rmse = []
    tuned_export_points = []
    base_rmse = None
    base_export_points = None
    for label, root in RUNS:
        base, tuned = load_metrics(root)
        labels.append(label)
        tuned_rmse.append(float(tuned["ba_final_rmse"]))
        tuned_export_points.append(int(tuned["export_points"]))
        base_rmse = float(base["ba_final_rmse"])
        base_export_points = int(base["export_points"])

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.bar(labels, tuned_rmse, color=["#ef4444", "#f59e0b", "#dc2626"], width=0.58)
    ax.axhline(base_rmse, color="#111827", linestyle="--", linewidth=2, label=f"frozen VGGT {base_rmse:.2f}")
    ax.set_ylabel("Data2 BA final RMSE (px)")
    ax.set_title("Data1 fine-tuning transfer to Data2: geometry")
    ax.set_ylim(0, max(tuned_rmse + [base_rmse]) * 1.22)
    ax.legend(frameon=False)
    for bar, value in zip(bars, tuned_rmse):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.06, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    geom_path = PPT / "vggt_data1_finetune_data2_geometry.png"
    fig.savefig(geom_path, dpi=220)
    plt.close(fig)

    conservative = ROOT / "outputs" / "vggt_data1_finetune_data2_transfer_conservative"
    base_render = json.loads((conservative / "data2_frozen_vggt_test_render_metrics.json").read_text(encoding="utf-8"))
    tuned_render = json.loads((conservative / "data2_data1_finetuned_test_render_metrics.json").read_text(encoding="utf-8"))
    names = ["frozen VGGT", "data1-finetuned"]
    psnr = [base_render["mean_psnr"], tuned_render["mean_psnr"]]
    ssim = [base_render["mean_ssim"], tuned_render["mean_ssim"]]
    fig, ax1 = plt.subplots(figsize=(7.4, 4.6))
    bars = ax1.bar(names, psnr, color=["#6b7280", "#f59e0b"], width=0.55)
    ax1.set_ylabel("Data2 held-out PSNR (dB)")
    ax1.set_ylim(0, max(psnr) * 1.35)
    for bar, value in zip(bars, psnr):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.2, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    ax2 = ax1.twinx()
    ax2.plot(names, ssim, color="#111827", marker="o", linewidth=2)
    ax2.set_ylabel("SSIM")
    ax2.set_ylim(0, max(ssim) * 1.8)
    for i, value in enumerate(ssim):
        ax2.text(i, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.set_title("Conservative data1 fine-tune: Data2 held-out render")
    fig.tight_layout()
    render_path = PPT / "vggt_data1_finetune_data2_render.png"
    fig.savefig(render_path, dpi=220)
    plt.close(fig)

    report = [
        "# Data1 Fine-tune to Data2 Transfer Summary",
        "",
        "| Variant | Data1 loss start | Data1 loss end | Data2 BA RMSE | Export points |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, root in RUNS:
        loss = json.loads((root / "data1_finetune" / "loss.json").read_text(encoding="utf-8"))
        _, tuned = load_metrics(root)
        report.append(
            f"| {label} | {loss[0]['loss']:.5f} | {loss[-1]['loss']:.5f} | {float(tuned['ba_final_rmse']):.3f} | {int(tuned['export_points'])} |"
        )
    report.extend(
        [
            "",
            f"Frozen VGGT baseline Data2 BA RMSE: {base_rmse:.3f}, export points: {base_export_points}.",
            "",
            "Conservative variant held-out render:",
            f"- Frozen VGGT: PSNR {base_render['mean_psnr']:.3f}, SSIM {base_render['mean_ssim']:.4f}",
            f"- Data1-finetuned: PSNR {tuned_render['mean_psnr']:.3f}, SSIM {tuned_render['mean_ssim']:.4f}",
            "",
            "Conclusion: this small data1 pseudo-label fine-tune overfits data1 camera geometry and does not transfer to data2.",
        ]
    )
    report_path = ROOT / "outputs" / "vggt_data1_finetune_transfer_summary.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(report_path)
    print(geom_path)
    print(render_path)


if __name__ == "__main__":
    main()
