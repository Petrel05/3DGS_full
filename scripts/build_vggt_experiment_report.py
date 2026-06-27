from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "vggt_data2_network_experiments"


GRAPHDECO_MODELS = {
    "baseline_p25": {
        "model": ROOT / "outputs" / "official_data2_graphdeco_30k_masked_clean_bg_ba_pruned",
        "render_metrics": OUT / "data2_baseline_pruned_render_metrics.json",
    },
    "masked_gray_input_p0": {
        "model": OUT / "masked_gray_input_p0_graphdeco_30k_strict_pruned",
        "render_metrics": OUT / "data2_masked_gray_render_metrics.json",
    },
    "foreground_crop_p0": {
        "model": OUT / "foreground_crop_p0_graphdeco_30k_strict_pruned",
        "render_metrics": OUT / "data2_vggt_crop_best_render_metrics.json",
    },
}


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def count_binary_ply_vertices(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                return None
            text = line.decode("ascii", errors="ignore").strip()
            if text.startswith("element vertex "):
                return int(text.split()[-1])
            if text == "end_header":
                return None


def count_text_ply_vertices(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line.strip() == "end_header":
                return None
    return None


def count_points3d(dataset: Path) -> int | None:
    path = dataset / "sparse" / "0" / "points3D.txt"
    if not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#"))


def round_or_none(value: object, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def main() -> None:
    summary_rows = load_json(OUT / "vggt_network_optimization_data2_summary.json")
    rows = []
    baseline = next(row for row in summary_rows if row["name"] == "baseline_p25")
    for row in summary_rows:
        item = {
            "experiment": row["name"],
            "method": row["description"],
            "ba_final_rmse": round_or_none(row["ba_final_rmse"], 3),
            "ba_rmse_delta_vs_baseline": round_or_none(row["ba_final_rmse"] - baseline["ba_final_rmse"], 3),
            "ba_rmse_improvement_pct": round_or_none((baseline["ba_final_rmse"] - row["ba_final_rmse"]) / baseline["ba_final_rmse"] * 100.0, 1),
            "sparse_points": row["sparse_points"],
            "pair_inliers_sum": row["pair_inliers_sum"],
            "strict_foreground_points": row["strict_foreground_points"],
        }
        graph = GRAPHDECO_MODELS.get(row["name"])
        if graph is not None:
            ply = graph["model"] / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            item["graphdeco_pruned_gaussians"] = count_binary_ply_vertices(ply)
            metrics_path = graph["render_metrics"]
            if metrics_path.exists():
                metrics = load_json(metrics_path)
                item["render_psnr"] = round_or_none(metrics.get("mean_psnr"), 3)
                item["render_ssim"] = round_or_none(metrics.get("mean_ssim"), 4)
                item["render_fg_mae"] = round_or_none(metrics.get("mean_fg_mae"), 5)
        rows.append(item)

    csv_path = OUT / "vggt_data2_experiment_report.csv"
    keys = [
        "experiment",
        "method",
        "ba_final_rmse",
        "ba_rmse_delta_vs_baseline",
        "ba_rmse_improvement_pct",
        "sparse_points",
        "pair_inliers_sum",
        "strict_foreground_points",
        "graphdeco_pruned_gaussians",
        "render_psnr",
        "render_ssim",
        "render_fg_mae",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# VGGT Related Experiments on Data2",
        "",
        "These experiments do not require SIBR viewer. They evaluate VGGT-side changes using BA geometry, foreground-filtered COLMAP points, Graphdeco model size, and offline render/GT metrics.",
        "",
        "## Main Quantitative Table",
        "",
        "| Experiment | VGGT-side change | BA final RMSE | Improvement | Sparse points | Pair inliers | Strict foreground points | 3DGS pruned gaussians | PSNR | SSIM | FG MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {method} | {ba_final_rmse} | {ba_rmse_improvement_pct}% | {sparse_points} | {pair_inliers_sum} | {strict_foreground_points} | {graphdeco_pruned_gaussians} | {render_psnr} | {render_ssim} | {render_fg_mae} |".format(
                **{key: row.get(key, "") for key in keys}
            )
        )
    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            "- Keeping all confidence points alone (`p0_strict_points`) did not improve sparse geometry on data2: BA RMSE stayed at 1.419 px.",
            "- Mask-guided VGGT input was effective: replacing background with neutral gray reduced BA RMSE from 1.419 px to 0.865 px, a 39.1% improvement.",
            "- Foreground-centric cropping gave the best geometry: BA RMSE reached 0.805 px, sparse points increased from 410 to 509, and pair inliers increased from 694 to 895.",
            "- Offline rendering shows a tradeoff: the original pipeline still has the highest training-view PSNR, while the VGGT-side variants improve camera/track geometry. This is useful evidence that geometry metrics and final 3DGS photometric metrics should both be reported.",
            "",
            "## Recommended PPT Claim",
            "",
            "VGGT initialization quality can be improved more effectively by changing the input distribution than by only changing confidence filtering. On data2, foreground-aware input preprocessing reduced BA reprojection error by 39-43%, but final 3DGS render quality still depends on downstream pruning/training choices.",
        ]
    )
    report_path = OUT / "vggt_data2_experiment_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)
    print(csv_path)


if __name__ == "__main__":
    main()
