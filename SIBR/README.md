# SIBR Viewer Bundle

这个目录保存用于最终实时可视化的 SIBR viewer 运行包。高斯训练在远程服务器完成，但服务器没有桌面环境，无法运行 SIBR 的实时窗口；因此 viewer 部分在另一台 Windows 机器上完成。当前包内的 viewer 是 Windows 可执行文件：

- `viewer/bin/SIBR_gaussianViewer_app.exe`
- `viewer/bin/SIBR_remoteGaussian_app.exe`
- `viewer/bin/*.dll`
- `viewer/resources/`
- `viewer/shaders/`

如果工作环境是 Linux，不能直接执行 `.exe`。最终展示视频基于 Windows 机器上的 SIBR viewer 运行结果。

推荐从项目根目录使用统一脚本生成或运行命令：

```bash
python scripts/run_sibr_viewer.py data1 --dry-run
```

在 Windows 机器上，如果项目目录结构保持一致，可运行：

```bash
python scripts/run_sibr_viewer.py data1
python scripts/run_sibr_viewer.py data2
python scripts/run_sibr_viewer.py scene128
```

如果模型或数据路径被移动，显式传入路径：

```bash
python scripts/run_sibr_viewer.py data2 ^
  --viewer-bin C:\path\to\SIBR_gaussianViewer_app.exe ^
  --model C:\path\to\official_data2_graphdeco_30k_masked_clean_bg ^
  --source C:\path\to\official_data2_colmap_50k_masked_clean ^
  --iteration 30000
```

Graphdeco 模型目录中的 `cfg_args` 记录了训练服务器上的绝对路径。跨机器运行时应显式传入当前 Windows 机器上的 `--model` 和 `--source`，不要依赖 `cfg_args` 里的旧路径。

`scripts/run_sibr_viewer.py` 只负责路径规整、文件存在性检查和命令封装，不改动官方 SIBR viewer 主逻辑。
