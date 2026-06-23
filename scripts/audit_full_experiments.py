from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "data1": {
        "source": ROOT / "大作业数据/数据1-人体",
        "final": ROOT / "outputs/final_data1_vggt200k",
        "verify": ROOT / "outputs/verify_data1_ba_gsplat_allviews",
    },
    "data2": {
        "source": ROOT / "大作业数据/数据2-人体",
        "final": ROOT / "outputs/final_data2_vggt200k",
        "verify": ROOT / "outputs/verify_data2_ba_gsplat_allviews",
    },
    "scene": {
        "source": ROOT / "大作业数据/数据3-场景_frames",
        "final": ROOT / "outputs/final_scene_vggt200k",
        "verify": ROOT / "outputs/verify_scene_ba_gsplat_allviews",
    },
}


def read_metrics(directory: Path) -> dict:
    return json.loads((directory / "metrics.json").read_text(encoding="utf-8"))


def count_images(source: Path) -> int:
    return len(sorted(source.glob("rgb_*.png")))


def require(condition: bool, message: str, failures: list[str]) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


def main() -> int:
    failures: list[str] = []
    print("Full experiment audit")
    print("=====================")
    for name, cfg in DATASETS.items():
        expected_images = count_images(cfg["source"])
        final = read_metrics(cfg["final"])
        verify = read_metrics(cfg["verify"])

        print(f"\n{name}: expected input views = {expected_images}")
        require(final["images"] == expected_images, f"VGGT final uses all views ({final['images']}/{expected_images})", failures)
        require(final["backend"] == "VGGT on cuda", f"VGGT final backend is CUDA ({final['backend']})", failures)
        require(final["gaussian_source"] == "vggt", f"VGGT final Gaussian source is VGGT dense ({final['gaussian_source']})", failures)
        require(final["gaussian_points"] >= 100_000, f"VGGT final has enough splats ({final['gaussian_points']} >= 100000)", failures)
        require((cfg["final"] / "viewer.html").exists(), f"VGGT final viewer exists ({cfg['final'] / 'viewer.html'})", failures)
        require((cfg["final"] / "vggt_initial_point_cloud.ply").exists(), "VGGT initial point cloud was exported", failures)

        require(verify["images"] == expected_images, f"BA/gsplat verification uses all views ({verify['images']}/{expected_images})", failures)
        require(verify["ba_backend"] == "scipy.optimize.least_squares", f"BA backend is SciPy least_squares ({verify['ba_backend']})", failures)
        require(verify["ba_final_rmse"] < verify["ba_initial_rmse"], f"BA RMSE improves ({verify['ba_initial_rmse']:.3f} -> {verify['ba_final_rmse']:.3f})", failures)
        require(verify["ba_final_rmse"] < 1.0, f"BA final RMSE is sub-pixel/near-sub-pixel ({verify['ba_final_rmse']:.3f} px < 1.0)", failures)
        require(verify["gaussian_backend"] == "gsplat-rasterization", f"Gaussian backend is gsplat ({verify['gaussian_backend']})", failures)
        require(verify["gaussian_final_loss"] < verify["gaussian_initial_loss"], f"gsplat loss improves ({verify['gaussian_initial_loss']:.6f} -> {verify['gaussian_final_loss']:.6f})", failures)
        require((cfg["verify"] / "viewer.html").exists(), f"BA/gsplat viewer exists ({cfg['verify'] / 'viewer.html'})", failures)

    print("\nResult:", "PASS" if not failures else "FAIL")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
