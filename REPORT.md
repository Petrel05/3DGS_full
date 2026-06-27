# 三维重建与 3D Gaussian Splatting 实验报告

## 1. 项目目标

输入为无相机标定参数的多视角图像或视频帧。项目需要完成：

1. 使用 VGGT 估计相机参数和初始点云。
2. 编程实现 Bundle Adjustment，优化相机外参和三维点。
3. 使用 3D Gaussian Splatting 优化高斯点云，并提供实时交互展示。

当前已完成的新版主流程采用 VGGT 初始化、SIFT tracks、Bundle Adjustment、BA 后 COLMAP 导出和官方 Graphdeco 3DGS 30k 训练。已完成结果包括：

- `outputs/official_data1_graphdeco_30k_masked_clean_bg_ba`
- `outputs/official_data2_graphdeco_30k_masked_clean_bg_ba`
- `outputs/official_scene48_graphdeco_30k_cropped_ba`

同时保留 `outputs/official_scene128_graphdeco_30k_cropped` 作为 128 帧 no-BA 高视角 baseline，用于比较帧数和 BA 接入带来的变化。
BA 效果分析另保留同帧数 no-BA 对照，包括 `outputs/official_data1_graphdeco_30k_masked_clean_bg`、`outputs/official_data2_graphdeco_30k_masked_clean_bg` 和 `outputs/official_scene48_graphdeco_30k_cropped_noba`。

由于高斯训练资源要求高，训练在远程服务器完成；但远程服务器没有桌面环境，无法进行实时可视化。因此最终 SIBR viewer 展示在另一台 Windows 机器上完成，相关代码和运行环境需求随项目一起提交。本仓库保留 `scripts/run_sibr_viewer.py` 统一整理路径和启动参数。

## 2. 方法

### 2.1 VGGT 初始化

`src/reconstruct3d/vggt_adapter.py` 调用 VGGT：

- `pose_enc` 转换为相机内外参。
- `world_points` 作为初始稠密点云。
- `world_points_conf` 用于置信度过滤。

项目在 VGGT 相机初始化下构建 SIFT sparse tracks，并在 BA 后写入 COLMAP text 格式，提供给官方 Graphdeco 3DGS 训练。人体数据的 mask 不直接输入 VGGT 网络，但会用于后续点云过滤和训练图像 alpha 处理。

### 2.2 Bundle Adjustment 编程实现

VGGT dense point map 不直接提供传统 BA 所需的跨图 2D 同名观测，所以项目使用 VGGT 相机作为初值，再用 SIFT 构建 sparse tracks。

BA 实现在 `src/reconstruct3d/ba.py`：

- 优化变量：除第一帧外的相机旋转、平移，以及三维点坐标。
- 损失：所有 track 的重投影残差。
- 优化器：`scipy.optimize.least_squares`。
- 鲁棒性：使用 `soft_l1` loss 降低错误匹配影响。

BA 后的相机和稀疏点被导出为 COLMAP text 数据，并接入官方 Graphdeco 3DGS 主训练；项目内部 `gsplat` 仍保留用于轻量验证和 BA 效果分析。

### 2.3 3D Gaussian Splatting

项目内部 `src/reconstruct3d/gaussian.py` 保留两类实现：

- `gsplat` 后端：调用 `gsplat.rendering.rasterization`，优化 Gaussian 位置、尺度、旋转、颜色和 opacity。
- fallback 后端：无 CUDA 时使用 PyTorch 自动微分做轻量多视图光度优化。

最终展示采用 BA 后 COLMAP 数据接入官方 Graphdeco 3DGS：

1. 用 VGGT 初始化相机和初始点云。
2. 在 VGGT 相机下构建 SIFT tracks，并通过 BA 优化相机外参和稀疏三维点。
3. 导出 BA 后 COLMAP text 数据集。
4. 官方 `train.py` 训练 30k iterations。
5. 官方 `render.py` 输出训练视角结果和 GT 对照。
6. 官方 SIBR viewer 展示训练好的高斯模型。

### 2.4 Mask、绿幕和场景裁剪

人体数据的主要问题是绿幕背景容易被训练成漂浮高斯。最终处理：

- `scripts/apply_alpha_masks_to_colmap_images.py` 将 RGB+mask 写成 RGBA PNG。
- `erode=1` 收缩前景，减少绿幕边缘进入训练。
- `feather=2` 平滑边界。
- `--despill-green` 压制绿色溢色。
- `--premultiply-rgb` 将 alpha 外 RGB 置黑。
- `scripts/graphdeco_alpha_bg_loss.patch` 给官方 `train.py` 增加 alpha 外背景约束。

场景视频是 16:9。VGGT 方图输入会产生上下白色 letterbox，因此使用 `scripts/crop_colmap_letterbox.py` 裁剪图像并同步更新 COLMAP 相机高度和 `cy`。新版 BA 主流程已完成 48 帧版本：

- `outputs/official_scene48_colmap_50k_cropped_ba`
- `outputs/official_scene48_graphdeco_30k_cropped_ba`

128 帧 no-BA 版本仍保留为高视角 baseline：

- `outputs/official_scene128_colmap_50k_cropped`
- `outputs/official_scene128_graphdeco_30k_cropped`

为了公平分析 BA 对 3DGS 的影响，另训练了同为 48 帧的 no-BA 对照：

- `outputs/official_scene48_colmap_50k_cropped_noba`
- `outputs/official_scene48_graphdeco_30k_cropped_noba`

## 3. 实验环境

- Conda 环境：`vggt`
- GPU：NVIDIA GeForce RTX 3090
- VGGT 权重：`.models/VGGT-1B/`
- 官方 Graphdeco 3DGS 源码：示例命令使用 `/tmp/gaussian-splatting-official`
- Python 依赖：见 `requirements.txt` 和 `requirements-vggt.txt`
- SIBR viewer：`SIBR/` 中有 Windows 运行包；远程服务器没有桌面环境，实时可视化在另一台 Windows 机器上运行

SIBR 相关脚本只封装命令，不修改官方 viewer 主逻辑；跨机器路径通过 `--viewer-bin`、`--model`、`--source` 参数覆盖。Graphdeco 输出中的 `cfg_args` 保留了训练服务器绝对路径，跨机器运行时必须显式指定当前机器上的模型目录和 COLMAP 输入目录。

## 4. 最终结果

| 数据 | 输入视角 | 官方 3DGS 输出 | 高斯数 | PLY 大小 | 说明 |
| --- | ---: | --- | ---: | ---: | --- |
| 数据1-人体 + BA | 16 | `outputs/official_data1_graphdeco_30k_masked_clean_bg_ba` | 39,531 | 9.4M | 新版主流程 |
| 数据2-人体 + BA | 16 | `outputs/official_data2_graphdeco_30k_masked_clean_bg_ba` | 62,854 | 15M | 新版主流程 |
| 数据3-场景 48帧 + BA | 48 | `outputs/official_scene48_graphdeco_30k_cropped_ba` | 919,502 | 218M | 新版主流程 |
| 数据3-场景 128帧 no-BA | 128 | `outputs/official_scene128_graphdeco_30k_cropped` | 988,158 | 234M | 高视角 baseline |

同帧数 BA/no-BA 对照结果：

| 数据 | 输入视角 | no-BA 输出 | with-BA 输出 | no-BA 高斯数 | with-BA 高斯数 | BA sparse RMSE |
| --- | ---: | --- | --- | ---: | ---: | --- |
| 数据1-人体 | 16 | `outputs/official_data1_graphdeco_30k_masked_clean_bg` | `outputs/official_data1_graphdeco_30k_masked_clean_bg_ba` | 63,048 | 39,531 | 1.633 -> 1.266 px |
| 数据2-人体 | 16 | `outputs/official_data2_graphdeco_30k_masked_clean_bg` | `outputs/official_data2_graphdeco_30k_masked_clean_bg_ba` | 63,339 | 62,854 | 1.821 -> 1.419 px |
| 数据3-场景 | 48 | `outputs/official_scene48_graphdeco_30k_cropped_noba` | `outputs/official_scene48_graphdeco_30k_cropped_ba` | 939,023 | 919,502 | 3.215 -> 1.289 px |

对应辅助 WebGL 预览：

- 数据1 BA：`outputs/official_data1_graphdeco_30k_masked_clean_bg_ba`
- 数据2 BA：`outputs/official_data2_graphdeco_30k_masked_clean_bg_ba`
- 场景48 BA：`outputs/official_scene48_graphdeco_30k_cropped_ba`

人体数据经过 BA、clean mask 和 alpha 背景损失后，大块绿色背景和漂浮绿幕高斯明显减少，主体边界更干净。场景 48 帧 BA 版本的最终高斯数约 92 万；同帧数 no-BA 版本约 94 万，128 帧 no-BA baseline 约 99 万。高斯数接近说明点数规模不是 BA 效果的唯一判断标准，视觉质量和相机几何一致性更关键。

## 5. 渲染图直观展示

当前机器不能运行 SIBR 实时窗口，因此报告中的静态直观展示使用官方 Graphdeco `render.py` 输出的训练视角渲染图，并用 `scripts/make_render_contact_sheet.py` 生成 render/GT 对照拼图。SIBR 视频用于最终交互展示，render/GT 拼图用于在无桌面服务器上检查训练质量。

| 数据 | render/GT 对照图 |
| --- | --- |
| 数据1-人体 + BA | `ppt_assets/render_gt_data1_ba_4views.png` |
| 数据2-人体 + BA | `ppt_assets/render_gt_data2_ba_4views.png` |
| 数据3-场景 48帧 + BA | `ppt_assets/render_gt_scene48_ba_4views.png` |
| 数据3-场景 48帧 no-BA | `ppt_assets/render_gt_scene48_noba_4views.png` |
| 数据3-场景 128帧 no-BA | `outputs/official_scene128_graphdeco_30k_cropped/train/ours_30000/render_vs_gt_contact_sheet.png` |

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

结论：在相同点数上限下，48 帧已经能得到接近 50k 的可见 VGGT 点云，适合快速实验，并作为当前 BA 主流程的场景版本；128 帧 no-BA 结果保留为高视角 baseline，用于说明更多视角覆盖能带来更完整的训练观察。

### 6.3 BA 接入主流程与同帧数对照

新版主流程已经将 BA 后 COLMAP 数据接入官方 Graphdeco 3DGS。BA 导出阶段的重投影误差如下：

| 数据 | 输入视角 | BA RMSE | BA 后 3DGS 输出 |
| --- | ---: | --- | --- |
| 数据1-人体 | 16 | 1.633 -> 1.266 px | `outputs/official_data1_graphdeco_30k_masked_clean_bg_ba` |
| 数据2-人体 | 16 | 1.821 -> 1.419 px | `outputs/official_data2_graphdeco_30k_masked_clean_bg_ba` |
| 数据3-场景 | 48 | 3.215 -> 1.289 px | `outputs/official_scene48_graphdeco_30k_cropped_ba` |

这说明 BA 在新版主流程中确实降低了 sparse tracks 的重投影误差，并将优化后的相机/稀疏点用于官方 3DGS 训练。

为了分析 BA 是否影响最终高斯泼溅，主报告采用同帧数、同预处理、同 Graphdeco 30k 训练设置的 no-BA/with-BA 对照：

| 数据 | no-BA 训练输入 | with-BA 训练输入 | no-BA 高斯数 | with-BA 高斯数 | 视觉对照素材 |
| --- | --- | --- | ---: | ---: | --- |
| 数据1-人体 16帧 | `outputs/official_data1_colmap_50k_masked_clean` | `outputs/official_data1_colmap_50k_masked_clean_ba` | 63,048 | 39,531 | `ppt_assets/render_gt_data1_4views.png` vs `ppt_assets/render_gt_data1_ba_4views.png` |
| 数据2-人体 16帧 | `outputs/official_data2_colmap_50k_masked_clean` | `outputs/official_data2_colmap_50k_masked_clean_ba` | 63,339 | 62,854 | `ppt_assets/render_gt_data2_4views.png` vs `ppt_assets/render_gt_data2_ba_4views.png` |
| 数据3-场景 48帧 | `outputs/official_scene48_colmap_50k_cropped_noba` | `outputs/official_scene48_colmap_50k_cropped_ba` | 939,023 | 919,502 | `ppt_assets/render_gt_scene48_noba_4views.png` vs `ppt_assets/render_gt_scene48_ba_4views.png` |

从数量上看，BA 后最终高斯数没有系统性增加：数据1 变少，数据2 基本持平，scene48 略少。这符合 3DGS 的训练机制：BA 主要改变相机位姿和稀疏几何一致性，Graphdeco 后续 densification/pruning 会根据光度残差自动增删高斯，因此高斯数只能作为模型规模参考，不能单独代表质量。更可靠的结论应来自同视角 render/GT 对照和 SIBR 截图：如果 BA 版本边界更稳定、漂浮点更少、视角切换更顺，则说明几何初始化对 3DGS 有正向作用。

PPT 使用 `ppt_assets/ba_same_frame_graphdeco_comparison.png` 汇总这三组主训练对照。

同时，为了对应作业中的 BA 和高斯优化编程实现，保留项目内部 `OpenCV -> BA -> gsplat` 轻量验证结果：

| 数据 | 输出目录 | BA RMSE | Gaussian loss |
| --- | --- | --- | --- |
| 数据1-人体 | `outputs/verify_data1_ba_gsplat_allviews` | 2.374 -> 0.410 px | 0.134112 -> 0.118603 |
| 数据2-人体 | `outputs/verify_data2_ba_gsplat_allviews` | 3.087 -> 0.873 px | 0.132482 -> 0.115660 |
| 数据3-场景 | `outputs/verify_scene_ba_gsplat_allviews` | 2.269 -> 0.681 px | 0.342095 -> 0.278719 |

轻量验证说明项目内部 BA 能显著降低重投影误差，`gsplat` 可微 rasterizer 的光度 loss 也能下降。最终展示采用官方 Graphdeco 3DGS，是因为其 densification、pruning 和 SH 表达质量更稳定。

### 6.4 输入预处理消融

已有结果还体现了两类重要工程改进：

- 人体：`masked_soft -> masked_clean_bg`，增加 despill、premultiply RGB 和 alpha 外背景约束，减少绿幕漂浮高斯。
- 场景：`official_scene128_graphdeco_30k -> official_scene128_graphdeco_30k_cropped`，裁掉 VGGT 方图 padding 产生的 letterbox，并同步修正 COLMAP 内参。

这些不修改 VGGT 网络本身，但改进了 VGGT 输出和官方 3DGS 输入之间的几何/图像一致性，属于提高重建精度的后处理优化。

### 6.5 data2 VGGT 输入/输出侧优化

为了进一步分析 VGGT 初始化质量，针对 data2 设计了多组 VGGT 侧实验。所有实验都在相同 data2 数据、相同后续 BA/Graphdeco 框架下比较，重点观察 sparse geometry 和离线 render/GT 指标。

| 实验 | 改动 | BA final RMSE | Sparse points | Pair inliers | 训练视角 PSNR |
| --- | --- | ---: | ---: | ---: | ---: |
| `baseline_p25` | confidence percentile 25 | 1.419 px | 410 | 694 | 35.742 dB |
| `p0_strict_points` | 保留所有 VGGT confidence 点 | 1.419 px | 410 | 694 | - |
| `masked_gray_input_p0` | VGGT 输入前将背景替换为灰色 | 0.865 px | 439 | 698 | 32.143 dB |
| `foreground_crop_p0` | 前景中心 crop，提高人体区域分辨率 | 0.805 px | 509 | 895 | 26.630 dB |
| `wide_tracks_loop_p0` | 更宽 sparse matching window + loop closure | 1.417 px | 408 | 694 | - |

对应报告和图表：

- `outputs/vggt_data2_network_experiments/vggt_data2_experiment_report.md`
- `ppt_assets/vggt_data2_ba_rmse_experiments.png`
- `ppt_assets/vggt_data2_render_metrics_experiments.png`

结论：单纯把 VGGT confidence 阈值降到 0 并不能改善 sparse 几何；更有效的是改变 VGGT 输入分布，例如背景置灰和前景 crop，可将 data2 的 BA RMSE 降低约 39%-43%。但这些几何改进没有稳定转化为最终 3DGS render PSNR，说明 VGGT sparse geometry 指标和最终 3DGS photometric 目标之间存在不一致。

### 6.6 8/4/4 held-out geometry adapter 实验

为了避免只看训练视角，设计了 data2 的 8/4/4 held-out 对照：

- test views 固定为 `[3, 7, 11, 15]`。
- 其余 12 张作为 train/validation pool。
- adapter 选择使用 3-fold 的 `8 train / 4 validation`。
- 最终 3DGS 训练只使用 12 张非 test 视角，4 张 test 只用于 held-out render 指标。

实验不是更新 VGGT 1B 主干，而是在 frozen VGGT 输出之后加入轻量 geometry adapter，优化相机和稀疏点的几何残差。

| 方法 | 几何指标 | Held-out PSNR | Held-out SSIM | FG MAE |
| --- | ---: | ---: | ---: | ---: |
| Frozen VGGT baseline | train-pool RMSE 1.488 px | 27.218 dB | 0.9589 | 0.00723 |
| Frozen VGGT + geometry adapter | refit RMSE 1.644 -> 1.249 px | 26.315 dB | 0.9527 | 0.00818 |

对应文件：

- `outputs/vggt_adapter_split_data2/summary.md`
- `outputs/vggt_adapter_split_data2/adapter_cross_validation_folds.csv`
- `ppt_assets/vggt_adapter_split_data2_cv_rmse.png`
- `ppt_assets/vggt_adapter_split_data2_render_metrics.png`

结论：geometry adapter 能降低 sparse 重投影误差，但没有提升 held-out render。该结果说明单独优化几何残差可能让训练相机/稀疏点更一致，却不一定提升 3DGS 的新视角泛化。

### 6.7 data1 小样本微调 VGGT camera-head adapter 后迁移 data2

为了验证“同类人体数据预训练/微调能否提高 VGGT 重建精度”，尝试使用 data1 的 BA 后 COLMAP 相机作为 pseudo-label，对 VGGT camera head 的尾部模块做小样本微调，再迁移到 data2。

微调范围没有动完整 1B backbone，只训练：

- `camera_head.pose_branch`
- `camera_head.poseLN_modulation`
- `camera_head.embed_pose`
- `camera_head.token_norm`
- `camera_head.trunk_norm`

同时尝试了更保守的 strong-anchor 和只训练 `pose_branch` 的版本。

| 变体 | Data1 loss | Data2 BA final RMSE | Export points |
| --- | ---: | ---: | ---: |
| Frozen VGGT baseline | - | 1.602 px | 49,955 |
| `pose_tail_80` | 0.01976 -> 0.00224 | 3.144 px | 42,561 |
| `pose_tail_30_anchor` | 0.01976 -> 0.01328 | 2.572 px | 45,946 |
| `pose_branch_80` | 0.01976 -> 0.01596 | 3.049 px | 47,137 |

对最保守版本继续训练 7k Graphdeco 并测试 data2 held-out views：

| 方法 | Held-out PSNR | Held-out SSIM |
| --- | ---: | ---: |
| Frozen VGGT | 10.956 dB | 0.0410 |
| data1-finetuned adapter | 10.701 dB | 0.0186 |

对应文件：

- `scripts/run_vggt_data1_finetune_data2_transfer.py`
- `outputs/vggt_data1_finetune_transfer_summary.md`
- `ppt_assets/vggt_data1_finetune_data2_geometry.png`
- `ppt_assets/vggt_data1_finetune_data2_render.png`

结论：data1 上的 pseudo-label loss 可以明显下降，但迁移到 data2 后几何和 render 指标均变差。这是一个负结果，说明单个 16-view 序列的小样本 VGGT camera-head 微调容易过拟合该序列相机几何，不能直接作为同类人体数据泛化增强。

### 6.8 8/4/4 VGGT / 3DGS joint pose adapter

进一步尝试把 VGGT 输出相机的校正放进 3DGS 训练目标里。实现方式是在 Graphdeco 训练中加入 per-train-view pose delta：

- VGGT 主干冻结。
- 3DGS 高斯正常训练。
- 每个训练视角有一个小的旋转/平移 delta。
- pose delta 通过 3DGS photometric loss 更新。
- test views `[3, 7, 11, 15]` 不参与 pose adapter 和 3DGS 训练，只用于最终测试。

由于官方 rasterizer 不直接对 view/projection matrix 回传梯度，实验中采用等价近似：在每个训练视角下对 Gaussian means 施加可微的 per-view pose transform，从而让 photometric loss 能更新 pose adapter。实现补丁保存在：

- `scripts/graphdeco_joint_pose_adapter.patch`

同一 8/4/4 split 下结果如下：

| 方法 | 优化目标 | Geometry RMSE | Held-out PSNR | Held-out SSIM | FG MAE | Gaussians |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Frozen VGGT baseline | 无 adapter | 1.488 px | 27.218 dB | 0.9589 | 0.00723 | 73,332 |
| Geometry adapter | sparse geometry loss | 1.249 px | 26.315 dB | 0.9527 | 0.00818 | 75,202 |
| Joint 3DGS pose adapter | 3DGS photometric loss | - | 25.122 dB | 0.9498 | 0.00896 | 84,597 |

joint pose adapter 的参数确实被优化：

- max rotation delta: 0.018293 rad
- mean rotation norm: 0.011710 rad
- max translation delta: 0.034602
- mean translation norm: 0.012806

对应文件：

- `outputs/vggt_adapter_split_data2/joint_844_report.md`
- `ppt_assets/vggt_joint_844_render_metrics.png`
- `ppt_assets/vggt_joint_844_gaussian_count.png`
- `ppt_assets/vggt_adapter_split_data2_joint_pose_test_render_sheet.png`

结论：joint pose adapter 确实由 photometric loss 更新，但 held-out render 质量低于 frozen VGGT baseline。它产生了更多高斯，说明训练视角拟合更强，但没有带来 test views 的泛化提升。在小视角人体数据上，render-loss-driven pose adaptation 容易过拟合训练视角。

## 7. 与作业要求的对应

### VGGT 相机和初始点云

`run_project.py --backend vggt` 调用 VGGT 得到相机和初始点云；新版主流程在 VGGT 相机下构建 SIFT tracks，执行 BA 后通过 `--export-colmap-dataset` 写出官方 3DGS 可读取的 COLMAP text 数据。

### BA 优化

`src/reconstruct3d/ba.py` 是项目内编程实现的 BA。它使用 SIFT tracks 构建多视角重投影残差，并联合优化外参和三维点。

### 3DGS 优化与实时展示

项目内部保留 `gsplat` 可微优化后端用于验证；最终效果采用 BA 后 COLMAP 数据训练官方 Graphdeco 3DGS 完整 30k。实时展示采用官方 SIBR viewer，运行脚本为 `scripts/run_sibr_viewer.py`。

### VGGT 改进方法调研与实验

报告中保留 confidence filtering、frame count sweep、mask/背景约束和 scene crop 消融。它们分别对应 VGGT confidence 后处理、输入帧数选择、mask-aware 点云/图像处理和 VGGT 输出几何修正，用于提高重建精度或重建速度。

## 8. SIBR 展示方式

远程训练服务器没有桌面环境，无法运行实时 SIBR viewer。将最终模型目录和对应 COLMAP 输入移动到 Windows SIBR 机器后，可以运行：

```bash
python scripts/run_sibr_viewer.py data1 \
  --viewer-bin /path/to/SIBR_gaussianViewer_app \
  --model /path/to/official_data1_graphdeco_30k_masked_clean_bg_ba \
  --source /path/to/official_data1_colmap_50k_masked_clean_ba \
  --iteration 30000
```

新版 BA 主流程预置名称：

- `data1`
- `data2`
- `scene48`

其中 `scene128` 仍保留为 128 帧 no-BA baseline 预置。

可先用 `--dry-run` 打印命令，确认 Windows SIBR 机器上的路径是否正确。三组数据需要的读取路径是 `-m/--model` 指向 Graphdeco 输出目录，`-s/--source` 指向对应 COLMAP 输入目录。

## 9. 局限

- 远程训练服务器没有桌面环境，无法实时可视化；SIBR viewer 的实际效果依赖 Windows 机器环境。
- SIBR 启动脚本只做路径规整，未对官方 viewer 行为做实验性修改。
- WebGL viewer 不是最终质量标准，主要用于在当前机器无法运行 SIBR 时快速检查空间分布。
- 人体数据存在姿态变化、遮挡和 mask 边缘误差，静态 3DGS 会受到一定影响。
