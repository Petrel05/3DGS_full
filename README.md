# 三维重建与 3D Gaussian Splatting

本项目完成从无标定多视角图像/视频到 3D Gaussian Splatting 的重建流程。当前最终采用的方法是：

1. 使用 VGGT 估计相机参数和初始点云。
2. 将 VGGT 结果导出为 COLMAP text 数据集，作为官方 Graphdeco 3DGS 的输入。
3. 对人体数据使用前景 mask、去绿幕溢色和 alpha 背景约束；对场景视频裁掉 VGGT 方图 padding 产生的上下白边。
4. 使用官方 Graphdeco 3DGS 训练 30k iterations。
5. 使用官方 SIBR viewer 展示最终高斯结果。由于高斯训练资源要求高，训练在远程服务器完成；但远程服务器没有桌面环境，无法进行实时可视化，因此 SIBR 部分在另一台 Windows 机器上完成，相关代码和运行环境需求会随项目一起提交。

## 最终采用结果

| 数据 | 最终 Graphdeco 输出 | 训练输入 | 高斯数 | 辅助 WebGL 预览 |
| --- | --- | --- | ---: | --- |
| 数据1-人体 | `outputs/official_data1_graphdeco_30k_masked_clean_bg` | `outputs/official_data1_colmap_50k_masked_clean` | 63,048 | `outputs/official_data1_graphdeco_30k_masked_clean_bg_viewer/viewer.html` |
| 数据2-人体 | `outputs/official_data2_graphdeco_30k_masked_clean_bg` | `outputs/official_data2_colmap_50k_masked_clean` | 63,339 | `outputs/official_data2_graphdeco_30k_masked_clean_bg_viewer/viewer.html` |
| 数据3-场景 | `outputs/official_scene128_graphdeco_30k_cropped` | `outputs/official_scene128_colmap_50k_cropped` | 988,158 | `outputs/official_scene128_graphdeco_30k_cropped_viewer/viewer.html` |

最终质量判断以官方 Graphdeco 训练输出、官方 `render.py`、以及 SIBR viewer 视频为准。项目内 WebGL viewer 只用于快速检查点云分布，它不等价于官方 SIBR viewer 的 SH 渲染。

补充的 VGGT confidence filtering、视频帧数选择、BA+gsplat 验证和输入预处理消融实验见 `REPORT.md`。

项目根目录的两个压缩包是给 SIBR 机器使用的数据包：

- `official_3dgs_sibr_data.tar.gz`：人体两组最终输出及对应 COLMAP 输入。
- `official_3dgs_sibr_data_cropped_scene.tar.gz`：包含裁剪后的场景最终输出及对应 COLMAP 输入。

`SIBR/` 目录保存 Windows viewer 运行包和环境说明，具体见 `SIBR/README.md`。

## 任务对应关系

| 作业要求 | 项目实现 |
| --- | --- |
| 使用 VGGT 求相机参数和初始点云 | `src/reconstruct3d/vggt_adapter.py` 调用 VGGT，读取 `pose_enc`、`world_points`、`world_points_conf`。 |
| 编程实现 BA 优化相机外参和点云 | `src/reconstruct3d/ba.py` 使用 SIFT tracks 和 `scipy.optimize.least_squares` 联合优化相机外参与三维点。 |
| 编程实现/调用 3DGS 并实时交互渲染 | `src/reconstruct3d/gaussian.py` 保留 `gsplat` 实现链路；最终展示采用官方 Graphdeco 3DGS + SIBR viewer。 |

旧的 VGGT dense viewer、BA+gsplat 验证链路仍保留在代码中，主要用于说明编程实现和对比验证；最终叙事以三组 official Graphdeco 30k 结果为主。

## 环境

Python 侧推荐使用已有 conda 环境 `vggt`：

```bash
conda run -n vggt python -c "import torch, cv2, scipy; print(torch.cuda.is_available())"
```

Python 依赖见：

- `requirements.txt`：本项目脚本的基础 Python 依赖。
- `requirements-vggt.txt`：VGGT、PyTorch、`gsplat` 等重依赖。

外部组件：

- VGGT 权重：`.models/VGGT-1B/`
- 官方 Graphdeco 3DGS 源码：默认示例使用 `/tmp/gaussian-splatting-official`
- 官方训练补丁：`scripts/graphdeco_alpha_bg_loss.patch`
- SIBR viewer：`SIBR/` 中有 Windows 运行包；远程服务器没有桌面环境，当前 Linux 本机也不能运行该 `.exe`，因此实时可视化在另一台 Windows 机器上完成。

## 复现最终方法

### 1. 导出 COLMAP text 数据

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

场景使用 128 帧视频抽帧：

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

场景视频经 VGGT 方图输入后会有上下 letterbox，训练前裁掉并同步修正 COLMAP 内参：

```bash
conda run -n vggt python scripts/crop_colmap_letterbox.py \
  --input outputs/official_scene128_colmap_50k \
  --output outputs/official_scene128_colmap_50k_cropped
```

### 2. 人体数据生成 clean RGBA

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

这一步使用 mask 写 RGBA、轻微收缩和羽化边缘、压制绿幕溢色，并把 alpha 外 RGB 压到黑色。

### 3. 训练官方 Graphdeco 3DGS

人体数据使用 clean RGBA 和 alpha 背景损失补丁后的官方 `train.py`：

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n vggt python /tmp/gaussian-splatting-official/train.py \
  -s /home/fhy/3DGS_full/outputs/official_data2_colmap_50k_masked_clean \
  -m /home/fhy/3DGS_full/outputs/official_data2_graphdeco_30k_masked_clean_bg \
  --iterations 30000 \
  --save_iterations 7000 15000 30000 \
  --test_iterations 7000 15000 30000 \
  --quiet
```

场景数据使用裁剪后的 COLMAP 输入，不使用 alpha mask：

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n vggt python /tmp/gaussian-splatting-official/train.py \
  -s /home/fhy/3DGS_full/outputs/official_scene128_colmap_50k_cropped \
  -m /home/fhy/3DGS_full/outputs/official_scene128_graphdeco_30k_cropped \
  --iterations 30000 \
  --save_iterations 7000 15000 30000 \
  --test_iterations 7000 15000 30000 \
  --quiet
```

### 4. 渲染、拼图和 WebGL 辅助预览

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

### 5. 在有 SIBR 的机器上运行 viewer

高斯训练在远程服务器完成，但服务器没有桌面环境，无法运行实时 SIBR viewer；当前 Linux 本机也不能运行 `SIBR/viewer/bin/SIBR_gaussianViewer_app.exe`。将最终输出目录和对应 COLMAP 输入传到 Windows SIBR 机器后，可用统一脚本启动：

```bash
python scripts/run_sibr_viewer.py data2 \
  --viewer-bin /path/to/SIBR_gaussianViewer_app \
  --model /path/to/official_data2_graphdeco_30k_masked_clean_bg \
  --source /path/to/official_data2_colmap_50k_masked_clean \
  --iteration 30000
```

如果目录结构与本项目一致，脚本会默认使用 `SIBR/viewer/bin/SIBR_gaussianViewer_app.exe`；也可以设置 viewer binary：

```bash
SIBR_VIEWER_BIN=/path/to/SIBR_gaussianViewer_app \
python scripts/run_sibr_viewer.py scene128 --dry-run
```

`--dry-run` 会打印实际命令，便于检查 Windows SIBR 机器上的路径。注意：Graphdeco 输出目录里的 `cfg_args` 记录的是训练服务器上的绝对路径；跨机器运行时不要依赖其中的旧路径，应显式传入当前机器上的 `--model` 和 `--source`。

## 关键脚本

- `run_project.py`：VGGT/BA/项目 Gaussian viewer 主流程，并可导出 COLMAP text 数据。
- `scripts/apply_alpha_masks_to_colmap_images.py`：人体数据 RGBA mask、去绿幕、边缘处理。
- `scripts/crop_colmap_letterbox.py`：裁掉 VGGT padding 产生的上下 letterbox，并同步修正 COLMAP 相机内参。
- `scripts/filter_colmap_points_by_alpha.py`：按 alpha 投影过滤 COLMAP 初始化点云的辅助工具。
- `scripts/graphdeco_alpha_bg_loss.patch`：官方 Graphdeco `train.py` 的 alpha 背景损失补丁。
- `scripts/convert_official_3dgs_to_viewer.py`：把官方 3DGS PLY 转成项目 WebGL 预览。
- `scripts/make_render_contact_sheet.py`：生成官方 render/GT 对照拼图。
- `scripts/run_sibr_viewer.py`：SIBR viewer 参数化启动脚本，解决跨机器路径不一致问题。

## 已知限制

- 远程训练服务器没有桌面环境，无法实时可视化；bundled Windows SIBR viewer 需要在 Windows 机器上运行。
- SIBR 启动脚本只做路径规整和命令封装，不修改官方 viewer 主逻辑。
- 人体数据存在姿态变化和遮挡，不是严格静态物体，静态 3DGS 会受到一定影响。
- WebGL 预览为近似检查工具，不能完全复现官方 SIBR viewer 的 SH 渲染。
