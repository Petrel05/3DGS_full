# 三维重建与 3D Gaussian Splatting 实验报告

## 1. 项目目标

输入为无相机标定参数的多视角图像或视频帧。项目需要完成：

1. 使用 VGGT 估计相机参数和初始点云。
2. 编程实现 Bundle Adjustment，优化相机外参和三维点。
3. 使用 3D Gaussian Splatting 优化高斯点云，并提供实时交互展示。

当前最终采用三组官方 Graphdeco 3DGS 30k 结果：

- `outputs/official_data1_graphdeco_30k_masked_clean_bg`
- `outputs/official_data2_graphdeco_30k_masked_clean_bg`
- `outputs/official_scene128_graphdeco_30k_cropped`

由于高斯训练资源要求高，训练在远程服务器完成；但远程服务器没有桌面环境，无法进行实时可视化。因此最终 SIBR viewer 展示在另一台 Windows 机器上完成，相关代码和运行环境需求随项目一起提交。本仓库保留 `scripts/run_sibr_viewer.py` 统一整理路径和启动参数。

## 2. 方法

### 2.1 VGGT 初始化

`src/reconstruct3d/vggt_adapter.py` 调用 VGGT：

- `pose_enc` 转换为相机内外参。
- `world_points` 作为初始稠密点云。
- `world_points_conf` 用于置信度过滤。

项目将 VGGT 输出写入 COLMAP text 格式，提供给官方 Graphdeco 3DGS 训练。人体数据的 mask 不直接输入 VGGT 网络，但会用于后续点云过滤和训练图像 alpha 处理。

### 2.2 Bundle Adjustment 编程实现

VGGT dense point map 不直接提供传统 BA 所需的跨图 2D 同名观测，所以项目使用 VGGT/OpenCV 相机作为初值，再用 SIFT 构建 tracks。

BA 实现在 `src/reconstruct3d/ba.py`：

- 优化变量：除第一帧外的相机旋转、平移，以及三维点坐标。
- 损失：所有 track 的重投影残差。
- 优化器：`scipy.optimize.least_squares`。
- 鲁棒性：使用 `soft_l1` loss 降低错误匹配影响。

这条链路保留用于满足“编程实现 BA”的要求；最终高质量展示采用官方 Graphdeco 3DGS 训练结果。

### 2.3 3D Gaussian Splatting

项目内部 `src/reconstruct3d/gaussian.py` 保留两类实现：

- `gsplat` 后端：调用 `gsplat.rendering.rasterization`，优化 Gaussian 位置、尺度、旋转、颜色和 opacity。
- fallback 后端：无 CUDA 时使用 PyTorch 自动微分做轻量多视图光度优化。

最终展示采用官方 Graphdeco 3DGS：

1. 用 VGGT 导出的相机和初始点云生成 COLMAP text 数据集。
2. 官方 `train.py` 训练 30k iterations。
3. 官方 `render.py` 输出训练视角结果和 GT 对照。
4. 官方 SIBR viewer 展示训练好的高斯模型。

### 2.4 Mask、绿幕和场景裁剪

人体数据的主要问题是绿幕背景容易被训练成漂浮高斯。最终处理：

- `scripts/apply_alpha_masks_to_colmap_images.py` 将 RGB+mask 写成 RGBA PNG。
- `erode=1` 收缩前景，减少绿幕边缘进入训练。
- `feather=2` 平滑边界。
- `--despill-green` 压制绿色溢色。
- `--premultiply-rgb` 将 alpha 外 RGB 置黑。
- `scripts/graphdeco_alpha_bg_loss.patch` 给官方 `train.py` 增加 alpha 外背景约束。

场景视频是 16:9。VGGT 方图输入会产生上下白色 letterbox，因此使用 `scripts/crop_colmap_letterbox.py` 裁剪图像并同步更新 COLMAP 相机高度和 `cy`，最终采用裁剪后的：

- `outputs/official_scene128_colmap_50k_cropped`
- `outputs/official_scene128_graphdeco_30k_cropped`

## 3. 实验环境

- Conda 环境：`vggt`
- GPU：NVIDIA GeForce RTX 3090
- VGGT 权重：`.models/VGGT-1B/`
- 官方 Graphdeco 3DGS 源码：示例命令使用 `/tmp/gaussian-splatting-official`
- Python 依赖：见 `requirements.txt` 和 `requirements-vggt.txt`
- SIBR viewer：`SIBR/` 中有 Windows 运行包；远程服务器没有桌面环境，实时可视化在另一台 Windows 机器上运行

SIBR 相关脚本只封装命令，不修改官方 viewer 主逻辑；跨机器路径通过 `--viewer-bin`、`--model`、`--source` 参数覆盖。Graphdeco 输出中的 `cfg_args` 保留了训练服务器绝对路径，跨机器运行时必须显式指定当前机器上的模型目录和 COLMAP 输入目录。

## 4. 最终结果

| 数据 | 输入视角 | 官方 3DGS 输出 | 高斯数 | PLY 大小 | render/GT 对照 |
| --- | ---: | --- | ---: | ---: | --- |
| 数据1-人体 | 16 | `outputs/official_data1_graphdeco_30k_masked_clean_bg` | 63,048 | 15M | `train/ours_30000/render_vs_gt_contact_sheet.png` |
| 数据2-人体 | 16 | `outputs/official_data2_graphdeco_30k_masked_clean_bg` | 63,339 | 15M | `train/ours_30000/render_vs_gt_contact_sheet.png` |
| 数据3-场景 | 128 | `outputs/official_scene128_graphdeco_30k_cropped` | 988,158 | 234M | `train/ours_30000/render_vs_gt_contact_sheet.png` |

对应辅助 WebGL 预览：

- 数据1：`outputs/official_data1_graphdeco_30k_masked_clean_bg_viewer/viewer.html`
- 数据2：`outputs/official_data2_graphdeco_30k_masked_clean_bg_viewer/viewer.html`
- 场景：`outputs/official_scene128_graphdeco_30k_cropped_viewer/viewer.html`

人体数据经过 clean mask 和 alpha 背景损失后，大块绿色背景和漂浮绿幕高斯明显减少，主体边界更干净。场景数据使用 128 帧训练，视角更多、结构更复杂，最终高斯数接近百万，因此文件远大于人体数据。

## 5. 渲染图直观展示

当前机器不能运行 SIBR 实时窗口，因此报告中的静态直观展示使用官方 Graphdeco `render.py` 输出的训练视角渲染图，并用 `scripts/make_render_contact_sheet.py` 生成 render/GT 对照拼图。SIBR 视频用于最终交互展示，render/GT 拼图用于在无桌面服务器上检查训练质量。

| 数据 | render/GT 对照图 |
| --- | --- |
| 数据1-人体 | `outputs/official_data1_graphdeco_30k_masked_clean_bg/train/ours_30000/render_vs_gt_contact_sheet.png` |
| 数据2-人体 | `outputs/official_data2_graphdeco_30k_masked_clean_bg/train/ours_30000/render_vs_gt_contact_sheet.png` |
| 数据3-场景 | `outputs/official_scene128_graphdeco_30k_cropped/train/ours_30000/render_vs_gt_contact_sheet.png` |

同时保留两组关键消融的 render/GT 对照：

- 人体 mask/背景约束前后：`outputs/official_data1_graphdeco_30k_masked_soft/train/ours_30000/render_vs_gt_contact_sheet.png` 与 `outputs/official_data1_graphdeco_30k_masked_clean_bg/train/ours_30000/render_vs_gt_contact_sheet.png`
- 场景裁剪前后：`outputs/official_scene128_graphdeco_30k/train/ours_30000/render_vs_gt_contact_sheet.png` 与 `outputs/official_scene128_graphdeco_30k_cropped/train/ours_30000/render_vs_gt_contact_sheet.png`

## 6. VGGT 改进与消融实验

### 6.1 Confidence filtering

VGGT 输出 `world_points_conf`，可用置信度过滤低质量点。对数据1-人体固定 16 视角、mask-aware 可见性过滤和 KNN 轻量 viewer，只改变 `--vggt-confidence-percentile`：

| 输出目录 | confidence percentile | VGGT 输入点数 | 过滤后高斯输入点数 | 轻量 viewer splats |
| --- | ---: | ---: | ---: | ---: |
| `outputs/exp_conf_data1_p0` | 0 | 353,599 | 353,599 | 950 |
| `outputs/exp_conf_data1_p25` | 25 | 265,199 | 265,199 | 950 |
| `outputs/exp_conf_data1_p50` | 50 | 176,800 | 176,800 | 950 |

结论：提高 confidence percentile 会稳定减少低置信初始点。最终 official Graphdeco 输入采用 percentile 25，是精度和点数之间的折中；percentile 50 更快、更干净，但可能损失细节。

### 6.2 Frame count sweep

场景视频可以通过减少输入帧数提高 VGGT 推理和后续训练速度。这里固定 `--vggt-confidence-percentile 25`、`--vggt-max-points 50000`、visible filtering 和 KNN viewer，对比 48 帧和 128 帧：

| 输出目录 | 输入帧数 | VGGT 点数上限 | 可见性过滤后点数 | 轻量 viewer splats |
| --- | ---: | ---: | ---: | ---: |
| `outputs/exp_scene48_vggt50k` | 48 | 50,000 | 49,854 | 950 |
| `outputs/official_scene128_export_tmp` | 128 | 50,000 | 49,823 | 950 |

结论：在相同点数上限下，48 帧已经能得到接近 50k 的可见 VGGT 点云，适合快速实验；最终场景仍采用 128 帧，是为了获得更完整的视角覆盖和更稳定的官方 3DGS 训练。

### 6.3 BA 与 gsplat 编程实现验证

为了对应作业中的 BA 和高斯优化编程实现，保留项目内部 `VGGT/OpenCV -> BA -> gsplat` 验证结果：

| 数据 | 输出目录 | BA RMSE | Gaussian loss |
| --- | --- | --- | --- |
| 数据1-人体 | `outputs/verify_data1_ba_gsplat_allviews` | 2.374 -> 0.410 px | 0.134112 -> 0.118603 |
| 数据2-人体 | `outputs/verify_data2_ba_gsplat_allviews` | 3.087 -> 0.873 px | 0.132482 -> 0.115660 |
| 数据3-场景 | `outputs/verify_scene_ba_gsplat_allviews` | 2.269 -> 0.681 px | 0.342095 -> 0.278719 |

这说明项目内部 BA 能显著降低重投影误差，`gsplat` 可微 rasterizer 的光度 loss 也能下降。最终展示采用官方 Graphdeco 3DGS，是因为其 densification、pruning 和 SH 表达质量更稳定。

### 6.4 输入预处理消融

已有结果还体现了两类重要工程改进：

- 人体：`masked_soft -> masked_clean_bg`，增加 despill、premultiply RGB 和 alpha 外背景约束，减少绿幕漂浮高斯。
- 场景：`official_scene128_graphdeco_30k -> official_scene128_graphdeco_30k_cropped`，裁掉 VGGT 方图 padding 产生的 letterbox，并同步修正 COLMAP 内参。

这些不修改 VGGT 网络本身，但改进了 VGGT 输出和官方 3DGS 输入之间的几何/图像一致性，属于提高重建精度的后处理优化。

## 7. 与作业要求的对应

### VGGT 相机和初始点云

`run_project.py --backend vggt` 调用 VGGT 得到相机和初始点云，并通过 `--export-colmap-dataset` 写出官方 3DGS 可读取的 COLMAP text 数据。

### BA 优化

`src/reconstruct3d/ba.py` 是项目内编程实现的 BA。它使用 SIFT tracks 构建多视角重投影残差，并联合优化外参和三维点。

### 3DGS 优化与实时展示

项目内部保留 `gsplat` 可微优化后端；最终效果采用官方 Graphdeco 3DGS 完整训练 30k。实时展示采用官方 SIBR viewer，运行脚本为 `scripts/run_sibr_viewer.py`。

### VGGT 改进方法调研与实验

报告中保留 confidence filtering、frame count sweep、mask/背景约束和 scene crop 消融。它们分别对应 VGGT confidence 后处理、输入帧数选择、mask-aware 点云/图像处理和 VGGT 输出几何修正，用于提高重建精度或重建速度。

## 8. SIBR 展示方式

远程训练服务器没有桌面环境，无法运行实时 SIBR viewer；当前 Linux 本机也不能直接运行 `SIBR/viewer/bin/SIBR_gaussianViewer_app.exe`。将最终模型目录和对应 COLMAP 输入移动到 Windows SIBR 机器后，可以运行：

```bash
python scripts/run_sibr_viewer.py data1 \
  --viewer-bin /path/to/SIBR_gaussianViewer_app \
  --model /path/to/official_data1_graphdeco_30k_masked_clean_bg \
  --source /path/to/official_data1_colmap_50k_masked_clean \
  --iteration 30000
```

三组预置名称：

- `data1`
- `data2`
- `scene128`

可先用 `--dry-run` 打印命令，确认 Windows SIBR 机器上的路径是否正确。三组数据需要的读取路径是 `-m/--model` 指向 Graphdeco 输出目录，`-s/--source` 指向对应 COLMAP 输入目录。

## 9. 局限

- 远程训练服务器没有桌面环境，无法实时可视化；SIBR viewer 的实际效果依赖 Windows 机器环境。
- SIBR 启动脚本只做路径规整，未对官方 viewer 行为做实验性修改。
- WebGL viewer 不是最终质量标准，主要用于在当前机器无法运行 SIBR 时快速检查空间分布。
- 人体数据存在姿态变化、遮挡和 mask 边缘误差，静态 3DGS 会受到一定影响。
