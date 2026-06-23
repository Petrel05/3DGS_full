# 三维重建与高斯泼溅项目

本项目面向“大作业”要求：输入无相机标定参数的多视角图像或视频帧，使用 VGGT 估计相机和初始点云，再通过 Bundle Adjustment 与 3D Gaussian Splatting 风格优化得到可交互查看的三维结果。

当前版本已经从“纯手写简化优化”升级为“调用成熟开源实现优先”的流程：

- VGGT：调用官方 `facebookresearch/vggt`，从 `pose_enc` 转换相机内外参，并读取 `world_points/world_points_conf` 作为初始点云。
- BA：默认调用开源 `scipy.optimize.least_squares` 做联合非线性最小二乘优化；保留原来的 OpenCV PnP + 数值 Gauss-Newton 作为 fallback。
- 3D Gaussian：CUDA 可用且安装 `gsplat` 时优先调用开源 `gsplat.rendering.rasterization` 做可微渲染优化；无 CUDA 时自动降级为 PyTorch 自动微分的多视图光度优化。
- 展示：输出 `viewer.html`，使用 WebGL 点精灵 shader 绘制半透明 Gaussian splats，支持拖拽旋转和滚轮缩放。

## 任务对应关系

| PPT 要求 | 本项目实现 |
| --- | --- |
| 提供场景多视角图像，无相机标定参数 | `src/reconstruct3d/io_utils.py` 支持图像目录和 MP4 均匀抽帧；无标定输入由 VGGT 或 OpenCV fallback 估计相机。 |
| 使用 VGGT 求相机参数和初步点云 | `src/reconstruct3d/vggt_adapter.py` 调用官方 VGGT，读取 `pose_enc`、`world_points`、`world_points_conf`，并输出 `vggt_initial_point_cloud.ply`。 |
| Bundle Adjustment 优化相机外参和点云 | `src/reconstruct3d/ba.py` 默认用 `scipy.optimize.least_squares` 联合优化相机外参和全部重建点；记录 BA 前后 RMSE。 |
| 3D Gaussian 优化与实时交互渲染 | `src/reconstruct3d/gaussian.py` 优先使用 `gsplat` 可微 rasterizer；无 CUDA 时自动使用 PyTorch 光度优化 fallback；`viewer.html` 使用 WebGL 实时交互查看。 |
| 调研 VGGT 改进方法 | 本 README 和 `REPORT.md` 总结 confidence-aware、mask-aware、keyframe/coarse-to-fine、半精度/GPU、gsplat 加速等改进方向。 |

## 环境

推荐使用已有的 conda 环境 `vggt`：

```bash
conda run -n vggt python -c "import vggt, scipy, gsplat; print('env ok')"
```

依赖文件：

```bash
conda run -n vggt pip install -r requirements-vggt.txt
```

当前检查结果：

- `vggt` 环境可导入 VGGT、SciPy、OpenCV、PyTorch 和 `gsplat 1.5.3`。
- 非沙箱运行 `conda run -n vggt python` 时，`torch.cuda.is_available()` 为 `True`，可看到 4 张 NVIDIA GeForce RTX 3090。
- `.models/VGGT-1B/` 下已有 VGGT 权重；VGGT-1B 和 gsplat 都建议在 CUDA 环境运行。

## 运行方式

评分主线，严格对应“VGGT 相机/初始点云 + BA 优化相机外参和点云 + gsplat 高斯优化 + 实时 viewer”：

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
  --dense-min-visible-views 1 \
  --gaussian-backend gsplat \
  --gaussian-max-points 0 \
  --gaussian-iters 80 \
  --gaussian-size 256 \
  --gaussian-views 4
```

数据 2 只需要把 `--source` 和 `--output` 改成：

```bash
--source 大作业数据/数据2-人体
--output outputs/graded_data2_vggt_ba_gsplat_50k
```

场景帧推荐先用 12 张抽帧：

```bash
--source 大作业数据/数据3-场景_frames
--output outputs/graded_scene_vggt_ba_gsplat_50k
--dense-camera-filter visible
```

这条主线会用 VGGT 相机作为初值，在相邻视角窗口内构建 SIFT tracks，再用 SciPy BA 优化相机外参和稀疏三维点；随后用 BA 后相机过滤 VGGT dense 点云，并调用开源 `gsplat` rasterizer 做多视图光度优化。

VGGT 主路径，适合 CUDA 环境：

```bash
conda run -n vggt python run_project.py \
  --backend vggt \
  --vggt-model .models/VGGT-1B \
  --source 大作业数据/数据1-人体 \
  --output outputs/final_data1_vggt200k \
  --skip-vggt-sparse \
  --ba-backend none \
  --vggt-confidence-percentile 25 \
  --vggt-max-points 200000 \
  --gaussian-source vggt \
  --gaussian-backend knn \
  --gaussian-max-points 200000
```

这条命令用于生成当前推荐展示的 VGGT dense viewer。当前采样策略是 view-balanced spatial sampling：先给每个输入视角分配点数配额，再在每个视角内部做空间分层采样，避免正面帧被背面/侧面帧挤掉。三类数据的正式输出保留在 `outputs/final_data1_vggt200k`、`outputs/final_data2_vggt200k`、`outputs/final_scene_vggt200k`。

如果需要更细致的高密度版本，把 `--vggt-max-points` 和 `--gaussian-max-points` 改成 `400000`，输出到 `outputs/final_*_vggt400k`。人体数据会继续经过 mask 和置信度过滤，因此实际 splats 数可能低于 400000。

如果需要明确展示“调用开源 3D Gaussian Splatting rasterizer 做光度优化”，可运行全视角 BA + gsplat 验证版本：

```bash
conda run -n vggt python run_project.py \
  --backend opencv \
  --source 大作业数据/数据1-人体 \
  --output outputs/verify_data1_ba_gsplat_allviews \
  --max-size 640 \
  --max-features 1800 \
  --ba-points 120 \
  --ba-iters 2 \
  --ba-backend scipy \
  --gaussian-backend gsplat \
  --gaussian-max-points 500 \
  --gaussian-iters 10 \
  --gaussian-size 256 \
  --gaussian-views 0
```

`final_*_vggt200k` 是高密度交互展示版，使用 Gaussian splatting 形式的 WebGL viewer，三类数据均使用全部视角并输出 190000 个 splats；`verify_*_ba_gsplat_allviews` 是真实 `gsplat-rasterization` 光度优化验证版，三类数据均使用全部视角，且 BA RMSE 和 Gaussian loss 都下降。

默认运行不再做人为运行时间裁剪：`--max-images 0` 使用全部图像，`--max-size 0` 使用原始分辨率，`--max-features 0` 使用 OpenCV 默认不限特征数，`--ba-points 0` 优化全部重建点，`--vggt-max-points 0` 保留全部 VGGT 点，`--gaussian-max-points 0` 优化全部 Gaussian 点，`--gaussian-size 0` 使用原始分辨率，`--gaussian-views 0` 使用全部视角。

可复现实验路径，数据 1：

```bash
conda run -n vggt python run_project.py \
  --backend opencv \
  --source 大作业数据/数据1-人体 \
  --output outputs/verify_data1_ba_gsplat_allviews \
  --max-size 640 \
  --max-features 1800 \
  --ba-points 120 \
  --ba-iters 2 \
  --ba-backend scipy \
  --gaussian-backend gsplat \
  --gaussian-max-points 500 \
  --gaussian-iters 10 \
  --gaussian-size 256 \
  --gaussian-views 0
```

数据 2：

```bash
conda run -n vggt python run_project.py \
  --backend opencv \
  --source 大作业数据/数据2-人体 \
  --output outputs/verify_data2_ba_gsplat_allviews \
  --max-size 640 \
  --max-features 1800 \
  --ba-points 120 \
  --ba-iters 2 \
  --ba-backend scipy \
  --gaussian-backend gsplat \
  --gaussian-max-points 500 \
  --gaussian-iters 10 \
  --gaussian-size 256 \
  --gaussian-views 0
```

查看结果：

```bash
python3 -m http.server 8765
```

然后访问：

- `http://127.0.0.1:8765/outputs/final_data1_vggt200k/viewer.html`
- `http://127.0.0.1:8765/outputs/final_data2_vggt200k/viewer.html`
- `http://127.0.0.1:8765/outputs/final_scene_vggt200k/viewer.html`
- `http://127.0.0.1:8765/outputs/final_data1_vggt400k/viewer.html`
- `http://127.0.0.1:8765/outputs/final_data2_vggt400k/viewer.html`
- `http://127.0.0.1:8765/outputs/final_scene_vggt400k/viewer.html`
- `http://127.0.0.1:8765/outputs/verify_data1_ba_gsplat_allviews/viewer.html`
- `http://127.0.0.1:8765/outputs/verify_data2_ba_gsplat_allviews/viewer.html`
- `http://127.0.0.1:8765/outputs/verify_scene_ba_gsplat_allviews/viewer.html`

## 算法流程

1. 数据读取：目录输入读取 `rgb_*.png`，自动匹配 `msk_*.png`；视频输入用 OpenCV 均匀抽帧。
2. VGGT 初始化：调用 `VGGT.from_pretrained(...)`，得到相机 pose encoding、稠密点云和置信度。
3. Track 桥接：VGGT dense point map 不直接给 BA 所需的跨图 2D 同名观测，因此在 VGGT 相机初值下用 SIFT 构建 tracks 并三角化稀疏点。
4. BA：用 SciPy `least_squares` 优化除第一帧外的相机外参和全部重建点，使用 `soft_l1` loss 增强鲁棒性。
5. Gaussian 优化：优先调用 `gsplat` 可微 rasterizer；如果没有 CUDA，则使用 PyTorch 自动微分 fallback，对多视图目标图优化 Gaussian 位置、颜色、半径和透明度。
6. 交互渲染：`viewer.html` 内嵌优化后的 Gaussian 参数，浏览器端 WebGL 实时 splatting。

## RGB 与 Mask 的使用方式

- RGB 图像是主输入：VGGT 使用 RGB 估计相机和 `world_points`；OpenCV/SIFT 使用 RGB 提取特征；点云和 Gaussian 的颜色也来自 RGB 采样。
- mask 图像是约束输入：如果目录里存在与 `rgb_0000.png` 对应的 `msk_0000.png`，项目会自动读取并二值化；如果没有 mask 且启用 mask，会用绿幕颜色自动生成 foreground mask。
- mask 主要用于 SIFT：`cv2.SIFT.detectAndCompute(frame.image, frame.mask)` 只在前景区域提取特征，减少背景点和边缘噪声。
- mask 也用于 VGGT dense 点云过滤：项目会把 mask 按 VGGT 的 518 输入几何同步 resize/pad，再过滤 `world_points`，避免背景点进入 Gaussian viewer。
- 当前 VGGT 模型本身不直接吃 mask；mask-aware 改进体现在 SIFT 前景约束、VGGT 点云过滤、Gaussian 初始化和后续可扩展的 confidence-weighted BA。
- VGGT viewer 已改为优先使用 mask-aware 的 dense 初始点云生成 Gaussian；推荐优先查看 `outputs/final_data1_vggt200k`、`outputs/final_data2_vggt200k` 和 `outputs/final_scene_vggt200k`。

## 当前推荐结果

| 输出目录 | 数据 | 视角使用 | Gaussian 点数 | 几何厚度指标 |
| --- | --- | ---: | ---: | ---: |
| `outputs/final_data1_vggt200k` | 数据1-人体 | 16/16 | 190000 | 0.2167 |
| `outputs/final_data2_vggt200k` | 数据2-人体 | 16/16 | 190000 | 0.1923 |
| `outputs/final_scene_vggt200k` | 数据3-场景帧 | 12/12 | 190000 | 0.1698 |
| `outputs/final_data1_vggt400k` | 数据1-人体 | 16/16 | 251939 | 0.2165 |
| `outputs/final_data2_vggt400k` | 数据2-人体 | 16/16 | 207334 | 0.1898 |
| `outputs/final_scene_vggt400k` | 数据3-场景帧 | 12/12 | 380000 | 0.1910 |

点数判断：三类 200k 正式 viewer 都远高于 100k splats，足够做实时交互展示和观察整体结构；400k 版本更细致，但浏览器加载会更慢。几何厚度指标来自 Gaussian 点云 PCA 的最薄主成分比例，三类数据都不是旧版单层薄片式输出。

## 最新实验结果

以下结果来自当前全视角验证输出。为了让 BA 和可微 Gaussian 优化可复现，验证实验限制了 BA 点数、Gaussian 点数和渲染分辨率，但没有限制输入视角数：数据1/数据2 使用全部 16 张图，场景使用全部 12 张抽帧。第一次运行 `gsplat` 会编译 CUDA 扩展，后续运行会复用缓存。

| 数据 | 图像数 | BA 后点数 | BA 初始 RMSE | BA 后 RMSE | BA 降低比例 | Gaussian loss | Gaussian 后端 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 数据1-人体 | 16/16 | 493 | 2.374 px | 0.410 px | 82.7% | 0.134112 -> 0.118603 | gsplat-rasterization |
| 数据2-人体 | 16/16 | 380 | 3.087 px | 0.873 px | 71.7% | 0.132482 -> 0.115660 | gsplat-rasterization |
| 数据3-场景帧 | 12/12 | 743 | 2.269 px | 0.681 px | 70.0% | 0.342095 -> 0.278719 | gsplat-rasterization |

误差判断：三类数据 BA 后 RMSE 全部低于 1 px，并且相对初始 BA RMSE 下降 70% 以上，可以认为相机外参和稀疏点云优化质量较好。`scripts/audit_full_experiments.py` 会逐项检查三类数据、全视角、点数、RMSE 和 gsplat loss。

## BA 对 Gaussian 的影响

BA 后重投影误差明显下降，说明相机和点云的一致性提高。对 Gaussian 优化的影响主要有三点：

1. 投影更准：Gaussian 光度优化依赖相机投影，BA 后同一个三维点在不同视角下更容易落到对应物体区域。
2. 颜色更稳：投影误差下降后，点的多视图颜色监督更一致，减少漂浮点和颜色污染。
3. 半径更合理：几何更紧时，Gaussian splat 不需要过大半径弥补错位，交互旋转时轮廓更稳定。

## VGGT 改进方法

1. confidence-aware：使用 `world_points_conf` 和 `depth_conf` 过滤低置信度点，减少遮挡、边界和弱纹理区域噪声。
2. mask-aware：人体数据有前景 mask，可限制 SIFT track、VGGT 点云采样和 Gaussian 光度 loss，只优化目标区域。
3. keyframe/coarse-to-fine：先选基线较大的关键帧跑 VGGT/BA，再逐步加入邻近帧，降低相似视角冗余和计算量。
4. 半精度与 CUDA 加速：VGGT 和 gsplat 都适合 CUDA/autocast；在 CUDA 机器上可把 `--gaussian-backend auto` 切到 gsplat，获得更接近标准 3DGS 的可微 rasterization。
5. 置信度加权 BA：后续可把 VGGT 置信度、mask 边界距离和重投影残差作为权重传入 BA，进一步提高鲁棒性。

## 输出文件

每次运行会生成：

- `point_cloud.ply`：BA 后点云。
- `vggt_initial_point_cloud.ply`：VGGT 初始点云，仅 VGGT 后端生成。
- `gaussians.json`：优化后的 Gaussian 参数。
- `viewer.html`：浏览器交互查看器。
- `metrics.json`：后端、RMSE、Gaussian loss 等指标。
- `summary.md`：单次运行摘要。

更完整的实验说明见 `REPORT.md` 和 `FULL_EXPERIMENTS.md`。
