# 全量实验与验收记录

本记录用于回答四个验收问题：

- 误差是否优良。
- 是否利用了所有视角图片。
- 点数是否足够。
- 是否包含三类数据的三个重建。

结论：当前输出满足目标。三类数据均已生成 VGGT dense 高点数交互 viewer；三类数据也都完成了全视角 BA + `gsplat-rasterization` 验证，BA 后 RMSE 均低于 1 px，Gaussian loss 均下降。

## 输入数据

| 数据 | 输入目录 | RGB 视角数 | Mask |
| --- | --- | ---: | --- |
| 数据1-人体 | `大作业数据/数据1-人体` | 16 | 16 |
| 数据2-人体 | `大作业数据/数据2-人体` | 16 | 16 |
| 数据3-场景 | `大作业数据/数据3-场景_frames` | 12 | 0 |

## 最终交互展示结果

这些输出使用 VGGT-1B CUDA 推理，读取 `pose_enc` 得到相机参数，读取 `world_points/world_points_conf` 得到初步 dense 点云。人体数据使用 mask-aware 点云过滤。输出的 `viewer.html` 是浏览器 WebGL Gaussian splatting 风格实时查看器。

| 数据 | 输出目录 | 使用视角 | VGGT 点输入 | Viewer splats | 厚度指标 |
| --- | --- | ---: | ---: | ---: | ---: |
| 数据1-人体 | `outputs/final_data1_vggt200k` | 16/16 | 200000 | 190000 | 0.2167 |
| 数据2-人体 | `outputs/final_data2_vggt200k` | 16/16 | 200000 | 190000 | 0.1923 |
| 数据3-场景 | `outputs/final_scene_vggt200k` | 12/12 | 200000 | 190000 | 0.1698 |
| 数据1-人体 | `outputs/final_data1_vggt400k` | 16/16 | 265199 | 251939 | 0.2165 |
| 数据2-人体 | `outputs/final_data2_vggt400k` | 16/16 | 218247 | 207334 | 0.1898 |
| 数据3-场景 | `outputs/final_scene_vggt400k` | 12/12 | 400000 | 380000 | 0.1910 |

点数评估：200k 版本的三个最终 viewer 都保留 190000 个 splats，超过本项目验收阈值 100000，足够进行实时交互展示和整体结构观察。400k 版本恢复了 KNN Gaussian 阶段“保留 95%、剪掉最稀疏 5% 点”的设置，以减少离群点和拖影；人体数据由于 mask 和置信度过滤，实际可用点数低于 400000。厚度指标为 Gaussian 点云 PCA 的最薄主成分比例，三类数据均明显高于旧版薄片式输出。

## BA 与 gsplat 全视角验证

VGGT dense point map 本身不提供传统 BA 所需的跨图 2D track，因此 BA 验证使用 OpenCV/SIFT 构建 tracks，再用 `scipy.optimize.least_squares` 优化相机外参和稀疏点云。Gaussian 验证使用开源 `gsplat.rendering.rasterization` 做可微光度优化。

| 数据 | 输出目录 | 使用视角 | BA 后点数 | BA RMSE | Gaussian loss |
| --- | --- | ---: | ---: | --- | --- |
| 数据1-人体 | `outputs/verify_data1_ba_gsplat_allviews` | 16/16 | 493 | 2.374 -> 0.410 px | 0.134112 -> 0.118603 |
| 数据2-人体 | `outputs/verify_data2_ba_gsplat_allviews` | 16/16 | 380 | 3.087 -> 0.873 px | 0.132482 -> 0.115660 |
| 数据3-场景 | `outputs/verify_scene_ba_gsplat_allviews` | 12/12 | 743 | 2.269 -> 0.681 px | 0.342095 -> 0.278719 |

误差评估：三类数据 BA 后 RMSE 全部低于 1 px，且相对 BA 初始 RMSE 下降 70% 以上；`gsplat` 光度 loss 也全部下降。因此当前误差表现可以认为较好。

## 复现实验命令

最终 VGGT dense viewer 示例，数据1：

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

BA + gsplat 全视角验证示例，数据1：

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

`--max-images` 未设置，等价于默认 `0`，因此使用全部输入视角。验证实验限制的是 BA 优化点数、Gaussian 点数和渲染分辨率，不限制视角数。

## 一键验收

运行：

```bash
conda run -n vggt python scripts/audit_full_experiments.py
```

当前结果为 `PASS`。脚本检查以下条件：

- 三类数据的最终 VGGT 输出都使用全部视角。
- 三类数据的最终 VGGT 输出都来自 `VGGT on cuda`。
- 三类数据的最终 viewer 都不少于 100000 个 splats。
- 三类数据都导出了 `vggt_initial_point_cloud.ply`。
- 三类数据的 BA + gsplat 验证都使用全部视角。
- 三类数据的 BA 后 RMSE 都低于 1 px 且相对初始值下降。
- 三类数据的 Gaussian 后端都是 `gsplat-rasterization`，并且光度 loss 下降。

## 查看方式

可以直接打开对应输出目录下的 `viewer.html`，或在项目根目录启动静态服务器：

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
