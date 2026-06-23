# 三维重建与 3D Gaussian Splatting 实验报告

## 1. 任务目标

输入为无相机标定参数的多视角图像或视频帧。项目需要完成三件事：

1. 使用 VGGT 估计相机参数和初始点云。
2. 编程实现 Bundle Adjustment，优化相机外参和三维点。
3. 编程实现或调用 3D Gaussian Splatting 进行高斯点云优化，并展示实时交互渲染。

最终项目采用“两条互补链路”：

- 评分链路：VGGT 初始化、SIFT tracks、SciPy BA、`gsplat` 可微 rasterizer、WebGL 交互 viewer。
- 展示链路：VGGT 导出 COLMAP text 初始化，调用官方 Graphdeco 3DGS 训练 30k，再用官方 `render.py` 和 WebGL viewer 展示。

## 2. 方法

### 2.1 VGGT 初始化

`src/reconstruct3d/vggt_adapter.py` 调用官方 VGGT：

- `pose_enc` 转换为相机内外参。
- `world_points` 作为稠密初始点云。
- `world_points_conf` 用于置信度过滤。

人体数据带前景 mask。VGGT 本身不直接输入 mask，但项目会把 mask 同步到 VGGT 的 518 输入几何，用于过滤 dense 点云和后续 Gaussian 初始化。

### 2.2 Bundle Adjustment

VGGT dense point map 不提供传统 BA 所需的跨图 2D 同名观测，所以项目使用 VGGT 相机作为初值，在相邻视角之间用 SIFT 构建 tracks 并三角化稀疏点。

BA 实现在 `src/reconstruct3d/ba.py`：

- 优化变量：除第一帧外的相机旋转/平移，以及三维点坐标。
- 损失：所有 track 的重投影残差。
- 优化器：`scipy.optimize.least_squares`。
- 鲁棒性：使用 `soft_l1` loss，降低错误匹配影响。

这满足“编程实现 Bundle Adjustment 优化相机外参和点云”的要求。

### 2.3 3D Gaussian Splatting

项目内部 `src/reconstruct3d/gaussian.py` 支持两类优化：

- `gsplat` 后端：调用开源 `gsplat.rendering.rasterization`，优化 Gaussian 位置、尺度、旋转、颜色和 opacity。
- fallback 后端：无 CUDA 时使用 PyTorch 自动微分做多视图光度优化。

最终高质量展示使用官方 Graphdeco 3DGS：

1. 用项目导出的 VGGT 相机和点云写成 COLMAP text 格式。
2. 官方 `train.py` 训练 30k iteration，包含 densification/pruning。
3. 官方 `render.py` 输出训练视角 splatting 图。
4. `scripts/convert_official_3dgs_to_viewer.py` 将官方 PLY 转成项目 WebGL viewer。

### 2.4 Mask 和绿幕处理

人体数据的主要问题是绿幕背景容易被训练成漂浮高斯。最终处理如下：

- `scripts/apply_alpha_masks_to_colmap_images.py` 将 RGB+mask 写成 RGBA PNG。
- 使用 `erode=1` 轻微收缩前景，避免把绿幕边缘纳入训练。
- 使用 `feather=2` 平滑边界。
- 使用 `--despill-green` 压制绿色溢色。
- 使用 `--premultiply-rgb` 将 alpha 外 RGB 置黑。

同时对官方 Graphdeco `train.py` 加了背景约束补丁，保存在 `scripts/graphdeco_alpha_bg_loss.patch`。原版代码只把 `render * alpha` 后再算 loss，alpha 外区域无梯度，漂浮高斯不会被惩罚；补丁额外要求 alpha 外 render 接近黑色。

## 3. 实验环境

- Conda 环境：`vggt`
- GPU：NVIDIA GeForce RTX 3090
- VGGT 权重：`.models/VGGT-1B/`
- 官方 Graphdeco 3DGS 源码：`/tmp/gaussian-splatting-official`
- Python 依赖：PyTorch、SciPy、OpenCV、Pillow、`gsplat`、官方 3DGS rasterizer 依赖

SIBR 官方 viewer 尝试过编译，但当前环境卡在 GitLab extlibs 下载，因此最终使用官方 `render.py` 作为标准效果展示，用 WebGL viewer 作为交互预览。

## 4. 最终实验结果

### 4.1 输出汇总

| 数据 | 输入视角 | 官方 3DGS 输出 | 高斯数 | PLY 大小 | 官方 render/GT 对照 |
| --- | ---: | --- | ---: | ---: | --- |
| 数据1-人体 | 16 | `outputs/official_data1_graphdeco_30k_masked_clean_bg` | 63,048 | 15M | `train/ours_30000/render_vs_gt_contact_sheet.png` |
| 数据2-人体 | 16 | `outputs/official_data2_graphdeco_30k_masked_clean_bg` | 63,339 | 15M | `train/ours_30000/render_vs_gt_contact_sheet.png` |
| 数据3-场景 | 128 | `outputs/official_scene128_graphdeco_30k` | 1,040,352 | 247M | `train/ours_30000/render_vs_gt_contact_sheet.png` |

对应 WebGL 预览：

- 数据1：`outputs/official_data1_graphdeco_30k_masked_clean_bg_viewer/viewer.html`
- 数据2：`outputs/official_data2_graphdeco_30k_masked_clean_bg_viewer/viewer.html`
- 场景：`outputs/official_scene128_graphdeco_30k_viewer/viewer.html`

场景输出超过一百万高斯，WebGL 预览加载较慢；展示时优先使用官方 render/GT 拼图。

### 4.2 结果解读

人体数据在未处理 alpha 外背景时，会出现明显绿幕高斯。clean mask 和背景损失后，官方 render 中大块绿色背景和绿色丝状漂浮物基本消失，人物主体边界明显更干净。

场景数据没有前景 mask，因此直接使用 128 帧视频抽帧训练官方 3DGS。由于视角数量多、场景内容复杂，最终 densification 到约 104 万高斯，文件明显大于人体数据。

官方 3DGS 的输出比项目内部 WebGL 近似 viewer 更适合作为质量判断依据，因为它使用官方 rasterizer、SH 表示、densification 和 pruning。WebGL viewer 主要用于快速旋转检查三维分布。

## 5. 与作业要求的对应

### VGGT 相机和初始点云

`run_project.py --backend vggt` 会调用 VGGT 得到相机和初始点云。官方 3DGS 训练输入中的 `cameras.txt/images.txt/points3D.txt` 也是由 VGGT 输出转换得到。

### BA 优化

项目提供了可运行的 VGGT+BA+gsplat 评分链路。BA 使用 VGGT 相机作为初值，用 SIFT tracks 形成多视角观测，并在 `src/reconstruct3d/ba.py` 中联合优化外参和点云。

### 3DGS 优化与展示

项目内部实现了 `gsplat` 可微优化后端；最终展示进一步使用官方 Graphdeco 3DGS 完整训练 30k。展示方式包括：

- 官方 `render.py` 输出训练视角 splatting 图。
- render/GT 对照拼图。
- 由官方 PLY 转换得到的 WebGL 交互预览。

## 6. 复现命令摘要

人体 clean mask：

```bash
conda run -n vggt python scripts/apply_alpha_masks_to_colmap_images.py \
  --source 大作业数据/数据2-人体 \
  --dataset outputs/official_data2_colmap_50k_masked_clean \
  --max-size 518 \
  --erode 1 \
  --feather 2 \
  --premultiply-rgb \
  --despill-green
```

官方 3DGS 训练：

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n vggt python /tmp/gaussian-splatting-official/train.py \
  -s /home/fhy/3DGS_full/outputs/official_data2_colmap_50k_masked_clean \
  -m /home/fhy/3DGS_full/outputs/official_data2_graphdeco_30k_masked_clean_bg \
  --iterations 30000 \
  --save_iterations 7000 15000 30000 \
  --test_iterations 7000 15000 30000 \
  --quiet
```

官方 render 和拼图：

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
```

## 7. 局限和后续改进

- SIBR viewer 未能在当前环境编译成功；后续可离线准备 extlibs 或换网络环境继续编译。
- 人体数据存在姿态变化和遮挡，不是严格静态物体，静态 3DGS 会受到一定影响。
- 当前 VGGT 只提供初始化，mask 不直接输入 VGGT 网络；后续可以尝试 mask-aware feature matching、置信度加权 BA 和更强的动态人体先验。
- WebGL viewer 是轻量预览，不能完全复现官方 viewer 的 SH 渲染和交互特性。
