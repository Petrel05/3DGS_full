# 三维重建与高斯泼溅实验报告

## 1. 任务目标

本项目输入多视角图像或视频帧，不假设已知相机标定参数。目标是估计相机参数和初始点云，进一步优化相机外参、点云和 Gaussian 参数，并提供可交互三维查看结果。

## 2. 方法

### 2.1 VGGT 初始化

项目在 `src/reconstruct3d/vggt_adapter.py` 中调用官方 `facebookresearch/vggt`：

- `VGGT.from_pretrained(...)` 加载模型。
- `pose_encoding_to_extri_intri(...)` 将 `pose_enc` 转换为相机外参和内参。
- `world_points` 和 `world_points_conf` 提供 VGGT 稠密初始点云和置信度。

VGGT 输出的 dense point map 不直接包含传统 BA 所需的跨图同名 2D 观测，因此项目使用 VGGT 相机作为初始化，再用 SIFT 构建 track 并三角化可优化的稀疏点云。

### 2.2 Bundle Adjustment

当前版本使用开源 `scipy.optimize.least_squares` 做 BA，优化变量包括：

- 除第一帧外的相机旋转向量和平移向量。
- 全部重建三维点坐标。

损失函数为所有 track 的重投影残差，并使用 `soft_l1` 鲁棒 loss。第一帧相机固定，用于消除整体 gauge 自由度。旧版 OpenCV PnP + 数值 Gauss-Newton 实现保留为 fallback。命令行中 `--ba-points 0` 表示优化全部点，这也是当前默认设置。

### 2.3 3D Gaussian 优化

Gaussian 阶段支持两种后端：

- CUDA 可用时：优先调用开源 `gsplat.rendering.rasterization`，用可微 Gaussian rasterizer 优化 Gaussian 位置、尺度、旋转、颜色和透明度。
- 无 CUDA 环境：自动降级为 PyTorch 自动微分的多视图光度优化，对目标视角优化 Gaussian 位置、颜色、半径和透明度。

最终 `viewer.html` 使用 WebGL 点精灵 shader 实时显示优化后的 Gaussian 点云，支持拖拽旋转和滚轮缩放。

### 2.4 运行规模

当前默认不做运行时间裁剪：

- `--max-images 0`：使用全部输入图像或视频帧。
- `--max-size 0`：使用原始图像分辨率。
- `--max-features 0`：SIFT 特征数不设人工上限。
- `--ba-points 0`：BA 优化全部重建点。
- `--vggt-max-points 0`：VGGT 初始点云不按数量截断。
- `--gaussian-max-points 0`：Gaussian 优化和 viewer 使用全部 Gaussian 点。
- `--gaussian-size 0`：Gaussian 光度优化使用原始分辨率目标图。
- `--gaussian-views 0`：Gaussian 光度优化使用全部视角。

### 2.5 RGB 与 mask 的职责

RGB 是主输入，负责 VGGT 相机/点云估计、SIFT 特征提取和点云颜色采样。mask 是约束输入，项目会自动匹配 `rgb_XXXX.png` 和 `msk_XXXX.png`；有 mask 时 SIFT 只在前景区域提取特征，没有 mask 时可用绿幕颜色自动生成 foreground mask。当前 VGGT 模型本身不直接输入 mask，但项目会将 mask 同步到 VGGT 的 518 输入几何，用于过滤 `world_points`，避免背景点进入 Gaussian 初始化和 viewer。

## 3. 实验环境

实验使用 conda 环境 `vggt`：

- VGGT 可导入。
- SciPy、OpenCV、PyTorch 可导入。
- 已安装 `gsplat 1.5.3`。
- 非沙箱运行 `conda run -n vggt python` 时，`torch.cuda.is_available()` 为 `True`，可看到 4 张 NVIDIA GeForce RTX 3090。
- 本次新增实验实际使用 `gsplat-rasterization` CUDA 后端；第一次运行 gsplat 编译 CUDA 扩展耗时约 86 秒。

## 4. 实验结果

### 4.1 当前全量验收结果

本轮重新做了三类数据的全视角实验，并用 `scripts/audit_full_experiments.py` 自动验收。当前验收结果为 `PASS`。

最终交互展示输出如下。三类数据都使用 VGGT-1B CUDA 推理，使用全部输入视角，并输出高点数 WebGL Gaussian splatting viewer：

| 数据 | 输出目录 | 使用视角 | VGGT 输入点 | Viewer splats | 厚度指标 |
| --- | --- | ---: | ---: | ---: | ---: |
| 数据1-人体 | `outputs/final_data1_vggt200k` | 16/16 | 200000 | 190000 | 0.2167 |
| 数据2-人体 | `outputs/final_data2_vggt200k` | 16/16 | 200000 | 190000 | 0.1923 |
| 数据3-场景 | `outputs/final_scene_vggt200k` | 12/12 | 200000 | 190000 | 0.1698 |

BA 与 `gsplat-rasterization` 验证如下。验证实验限制的是 BA 点数、Gaussian 点数和渲染分辨率，不限制输入视角；三类数据都使用全部视角。BA 后 RMSE 全部低于 1 px，且 Gaussian photometric loss 全部下降。

| 数据 | 输出目录 | 使用视角 | BA 后点数 | BA RMSE | Gaussian loss |
| --- | --- | ---: | ---: | --- | --- |
| 数据1-人体 | `outputs/verify_data1_ba_gsplat_allviews` | 16/16 | 493 | 2.374 -> 0.410 px | 0.134112 -> 0.118603 |
| 数据2-人体 | `outputs/verify_data2_ba_gsplat_allviews` | 16/16 | 380 | 3.087 -> 0.873 px | 0.132482 -> 0.115660 |
| 数据3-场景 | `outputs/verify_scene_ba_gsplat_allviews` | 12/12 | 743 | 2.269 -> 0.681 px | 0.342095 -> 0.278719 |

这些结果直接回答验收问题：误差优良，全部视角已被使用，最终展示点数足够，并且包含数据1、数据2、数据3三类数据的三个重建结果。完整命令和验收项见 `FULL_EXPERIMENTS.md`。

### 4.2 结果解读

BA 明显降低重投影误差，说明相机外参和点云几何一致性得到提升。使用 `gsplat-rasterization` 后，Gaussian 光度 loss 也下降，说明优化后的颜色、尺度、旋转和 opacity 更符合输入视角图像。

针对 viewer 看起来“只有一层”的问题，当前实现采用 confidence-aware + view-balanced spatial sampling。旧版只按置信度取前若干点，容易保留最稳定的一层可见表面；当前版本先给每个输入视角分配配额，再做空间分层采样，并用 KDTree 替代大点云二次复杂度 KNN 半径估计。完整 16 张人体图和 12 张场景帧都已参与最终输出，点云厚度指标也证明结果不再是单层薄片式 viewer。

关于 3D Gaussian Splatting：`outputs/final_*_vggt200k` 是高密度交互展示版，使用 WebGL Gaussian splatting 形式渲染；`outputs/verify_*_ba_gsplat_allviews` 是真实 `gsplat-rasterization` 光度优化验证版。由于 dense VGGT 点云达到十万级，当前更适合将前者作为视觉展示，将后者作为“调用开源 3DGS rasterizer 优化”的证据。

## 5. BA 对 Gaussian 结果的影响

BA 优化会降低同名观测的重投影误差。Gaussian 优化依赖相机投影将三维 Gaussian 映射到图像平面，因此相机和点云越准确，光度优化越稳定。实验中 BA 后 RMSE 下降后，Gaussian 光度 loss 进一步下降，说明几何优化和 Gaussian 参数优化是互相促进的。

## 6. VGGT 改进方案

1. Confidence-aware + spatial sampling：使用 `world_points_conf` 过滤低置信度点，同时用空间分层采样避免只保留单层高置信表面。
2. Mask-aware reconstruction：人体数据使用前景 mask，避免绿幕背景和边缘错误点进入 BA、VGGT dense 点云和 Gaussian 初始化。
3. Coarse-to-fine keyframes：先选少量基线较大的关键帧估计稳定相机，再逐步加入相邻帧，提高速度并减少冗余。
4. CUDA/half precision：在 CUDA 机器上对 VGGT 使用 autocast 半精度，对 Gaussian 使用 gsplat rasterizer，提高速度并更接近标准 3DGS。
5. Confidence-weighted BA：将 VGGT 置信度、mask 边界距离和重投影残差结合为权重，进一步提升鲁棒性。

## 7. 局限

- WebGL viewer 是浏览器端交互查看器，不等价于论文原版 3DGS 的 SIBR/OpenGL viewer，但已经不再是 CPU Canvas 绘制。
- 人体数据存在非刚体变化，静态多视角重建会受到姿态变化和遮挡影响。
- 若要从“可见表面点云”进一步变成更完整的连续三维表面，需要加入 VGGT depth fusion/TSDF 融合、真实 3DGS densification/pruning 或人体先验模型；单纯调 viewer 不能补出输入中没有稳定观测到的背面几何。
