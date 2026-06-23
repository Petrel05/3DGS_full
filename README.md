# 三维重建与 3D Gaussian Splatting

本项目面向“大作业”要求：输入无相机标定参数的多视角图像或视频，完成 VGGT 相机/初始点云估计、Bundle Adjustment、3D Gaussian Splatting 优化，并提供可查看结果。

## 最终结果

当前推荐查看官方 Graphdeco 3DGS 30k 结果。人体数据使用前景 mask、去绿幕溢色和 alpha 背景约束；场景数据使用 128 帧视频抽帧。

| 数据 | 官方 PLY | 高斯数 | WebGL 预览 | 官方 render/GT 对照 |
| --- | --- | ---: | --- | --- |
| 数据1-人体 | `outputs/official_data1_graphdeco_30k_masked_clean_bg/point_cloud/iteration_30000/point_cloud.ply` | 63,048 | `outputs/official_data1_graphdeco_30k_masked_clean_bg_viewer/viewer.html` | `outputs/official_data1_graphdeco_30k_masked_clean_bg/train/ours_30000/render_vs_gt_contact_sheet.png` |
| 数据2-人体 | `outputs/official_data2_graphdeco_30k_masked_clean_bg/point_cloud/iteration_30000/point_cloud.ply` | 63,339 | `outputs/official_data2_graphdeco_30k_masked_clean_bg_viewer/viewer.html` | `outputs/official_data2_graphdeco_30k_masked_clean_bg/train/ours_30000/render_vs_gt_contact_sheet.png` |
| 数据3-场景 | `outputs/official_scene128_graphdeco_30k/point_cloud/iteration_30000/point_cloud.ply` | 1,040,352 | `outputs/official_scene128_graphdeco_30k_viewer/viewer.html` | `outputs/official_scene128_graphdeco_30k/train/ours_30000/render_vs_gt_contact_sheet.png` |

说明：WebGL 预览是便捷交互查看器，不等价于官方 SIBR viewer；场景 128 帧的 WebGL 文件较大，打开会比较慢。官方 `render.py` 生成的训练视角 splatting 图更适合快速展示最终效果。

## 任务对应关系

| 要求 | 实现位置 |
| --- | --- |
| 使用 VGGT 求相机参数和初步点云 | `src/reconstruct3d/vggt_adapter.py` 调用 VGGT，读取 `pose_enc`、`world_points`、`world_points_conf`。 |
| 编程实现 BA 优化相机外参和点云 | `src/reconstruct3d/ba.py` 用 `scipy.optimize.least_squares` 联合优化相机外参和三维点；VGGT 相机用于初始化，SIFT tracks 提供 BA 观测。 |
| 编程实现 3DGS 优化和实时交互渲染 | `src/reconstruct3d/gaussian.py` 支持 `gsplat` 可微 rasterizer；最终实验还调用官方 Graphdeco 3DGS 训练 30k，并用 WebGL viewer/官方 render 展示。 |

## 环境

推荐使用已有 conda 环境 `vggt`：

```bash
conda run -n vggt python -c "import torch, vggt, scipy, cv2; print(torch.cuda.is_available())"
```

已用到的主要组件：

- VGGT 权重：`.models/VGGT-1B/`
- 项目主流程：`run_project.py`
- 官方 Graphdeco 3DGS：`/tmp/gaussian-splatting-official`
- 官方训练补丁：`scripts/graphdeco_alpha_bg_loss.patch`

## 复现流程

### 1. 导出官方 COLMAP text 输入

以数据2为例：

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n vggt python run_project.py \
  --backend vggt \
  --vggt-model .models/VGGT-1B \
  --source 大作业数据/数据2-人体 \
  --output outputs/official_data2_export_tmp \
  --max-images 0 \
  --skip-vggt-sparse \
  --ba-backend none \
  --vggt-confidence-percentile 25 \
  --vggt-max-points 50000 \
  --gaussian-source vggt \
  --dense-camera-filter mask \
  --gaussian-backend knn \
  --gaussian-max-points 1000 \
  --export-colmap-dataset outputs/official_data2_colmap_50k \
  --outlier-percentile 100
```

场景 128 帧：

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n vggt python run_project.py \
  --backend vggt \
  --vggt-model .models/VGGT-1B \
  --source 大作业数据/数据3-场景.mp4 \
  --output outputs/official_scene128_export_tmp \
  --max-images 128 \
  --skip-vggt-sparse \
  --ba-backend none \
  --no-masks \
  --vggt-confidence-percentile 25 \
  --vggt-max-points 50000 \
  --gaussian-source vggt \
  --dense-camera-filter visible \
  --gaussian-backend knn \
  --gaussian-max-points 1000 \
  --export-colmap-dataset outputs/official_scene128_colmap_50k \
  --outlier-percentile 100
```

### 2. 人体数据生成 clean RGBA 图像

```bash
rsync -a outputs/official_data2_colmap_50k/ outputs/official_data2_colmap_50k_masked_clean/

conda run -n vggt python scripts/apply_alpha_masks_to_colmap_images.py \
  --source 大作业数据/数据2-人体 \
  --dataset outputs/official_data2_colmap_50k_masked_clean \
  --max-size 518 \
  --erode 1 \
  --feather 2 \
  --premultiply-rgb \
  --despill-green
```

这一步会轻微收缩 mask、羽化边缘、压制绿幕溢色，并把 alpha 外 RGB 置黑。它解决了官方 3DGS 学到绿幕漂浮高斯的问题。

### 3. 训练官方 3DGS

人体数据使用 clean RGBA 输入和背景约束补丁后的官方 `train.py`：

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n vggt python /tmp/gaussian-splatting-official/train.py \
  -s /home/fhy/3DGS_full/outputs/official_data2_colmap_50k_masked_clean \
  -m /home/fhy/3DGS_full/outputs/official_data2_graphdeco_30k_masked_clean_bg \
  --iterations 30000 \
  --save_iterations 7000 15000 30000 \
  --test_iterations 7000 15000 30000 \
  --quiet
```

场景数据不使用 alpha mask：

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n vggt python /tmp/gaussian-splatting-official/train.py \
  -s /home/fhy/3DGS_full/outputs/official_scene128_colmap_50k \
  -m /home/fhy/3DGS_full/outputs/official_scene128_graphdeco_30k \
  --iterations 30000 \
  --save_iterations 7000 15000 30000 \
  --test_iterations 7000 15000 30000 \
  --quiet
```

### 4. 渲染、拼图和 WebGL 预览

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n vggt python /tmp/gaussian-splatting-official/render.py \
  -s /home/fhy/3DGS_full/outputs/official_data2_colmap_50k_masked_clean \
  -m /home/fhy/3DGS_full/outputs/official_data2_graphdeco_30k_masked_clean_bg \
  --iteration 30000 \
  --skip_test \
  --quiet

conda run -n vggt python scripts/make_render_contact_sheet.py \
  --render-dir outputs/official_data2_graphdeco_30k_masked_clean_bg/train/ours_30000/renders \
  --gt-dir outputs/official_data2_graphdeco_30k_masked_clean_bg/train/ours_30000/gt \
  --output outputs/official_data2_graphdeco_30k_masked_clean_bg/train/ours_30000/render_vs_gt_contact_sheet.png

conda run -n vggt python scripts/convert_official_3dgs_to_viewer.py \
  --input outputs/official_data2_graphdeco_30k_masked_clean_bg/point_cloud/iteration_30000/point_cloud.ply \
  --output outputs/official_data2_graphdeco_30k_masked_clean_bg_viewer
```

## BA/gsplat 评分主线

如果需要单独展示“VGGT 相机 -> BA -> gsplat”这一评分链路，可运行：

```bash
conda run -n vggt python run_project.py \
  --backend vggt \
  --vggt-model .models/VGGT-1B \
  --source 大作业数据/数据1-人体 \
  --output outputs/graded_data1_vggt_ba_gsplat_50k \
  --max-features 3000 \
  --ba-backend scipy \
  --ba-points 0 \
  --ba-iters 8 \
  --vggt-sparse-match-window 2 \
  --vggt-sparse-loop-closure \
  --vggt-confidence-percentile 25 \
  --vggt-max-points 50000 \
  --gaussian-source vggt \
  --dense-camera-filter mask \
  --gaussian-backend gsplat \
  --gaussian-max-points 0 \
  --gaussian-iters 80 \
  --gaussian-size 256 \
  --gaussian-views 4
```

## 关键脚本

- `run_project.py`：VGGT/BA/项目 Gaussian viewer 主流程，并可导出 COLMAP text 数据。
- `scripts/apply_alpha_masks_to_colmap_images.py`：人体数据 RGBA mask、去绿幕、边缘处理。
- `scripts/filter_colmap_points_by_alpha.py`：按 alpha 投影过滤 COLMAP 初始化点云的辅助工具。
- `scripts/graphdeco_alpha_bg_loss.patch`：官方 Graphdeco `train.py` 的 alpha 背景损失补丁。
- `scripts/convert_official_3dgs_to_viewer.py`：把官方 3DGS PLY 转成项目 WebGL 预览。
- `scripts/make_render_contact_sheet.py`：生成官方 render/GT 对照拼图。

## 已知限制

- 官方 SIBR viewer 依赖下载在当前环境中卡在 GitLab extlibs，尚未编译成功。
- WebGL 预览是近似交互查看，最终质量判断以官方 `render.py` 输出为主。
- 人体数据不是严格静态物体，遮挡、姿态变化和 mask 边缘都会影响重建质量。
