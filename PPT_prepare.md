## 最新主流程更新（BA 接入 Graphdeco 主训练）

当前答辩主线应改为：

```text
Input images / scene frames
  -> VGGT-1B initialization
  -> SIFT sparse tracks under VGGT cameras
  -> Bundle Adjustment
  -> BA-refined COLMAP text export
  -> data preprocessing
  -> Graphdeco 3DGS 30k
  -> SIBR real-time viewer
```

已经完成的新版 BA 主流程结果：

| 数据 | 输入视角 | Graphdeco 输出 | 训练输入 | 高斯数 | PLY 大小 |
| --- | ---: | --- | --- | ---: | ---: |
| 数据1-人体 + BA | 16 | `outputs/official_data1_graphdeco_30k_masked_clean_bg_ba` | `outputs/official_data1_colmap_50k_masked_clean_ba` | 39,531 | 9.4M |
| 数据2-人体 + BA | 16 | `outputs/official_data2_graphdeco_30k_masked_clean_bg_ba` | `outputs/official_data2_colmap_50k_masked_clean_ba` | 62,854 | 15M |
| 数据3-场景 48帧 + BA | 48 | `outputs/official_scene48_graphdeco_30k_cropped_ba` | `outputs/official_scene48_colmap_50k_cropped_ba` | 919,502 | 218M |

同帧数 no-BA/with-BA 对照也已经完成，BA 效果分析页应使用这三组，而不是只用 scene：

| 数据 | no-BA Graphdeco 输出 | with-BA Graphdeco 输出 | no-BA 高斯数 | with-BA 高斯数 | BA sparse RMSE |
| --- | --- | --- | ---: | ---: | --- |
| 数据1-人体 16帧 | `outputs/official_data1_graphdeco_30k_masked_clean_bg` | `outputs/official_data1_graphdeco_30k_masked_clean_bg_ba` | 63,048 | 39,531 | 1.633 -> 1.266 px |
| 数据2-人体 16帧 | `outputs/official_data2_graphdeco_30k_masked_clean_bg` | `outputs/official_data2_graphdeco_30k_masked_clean_bg_ba` | 63,339 | 62,854 | 1.821 -> 1.419 px |
| 数据3-场景 48帧 | `outputs/official_scene48_graphdeco_30k_cropped_noba` | `outputs/official_scene48_graphdeco_30k_cropped_ba` | 939,023 | 919,502 | 3.215 -> 1.289 px |

保留 `outputs/official_scene128_graphdeco_30k_cropped` 作为 128 帧 no-BA 高视角 baseline，不再把它描述为最新版主流程。

### 已做 PPT 页面需要替换的地方

- “项目目标与完成情况”页：把 3DGS 实现方式改成“BA 后 COLMAP -> Graphdeco 3DGS 30k -> SIBR”，不要只写 VGGT 直接导出 COLMAP。
- “总体 Pipeline”页：BA 从下方验证支路移动到主流程中间；`gsplat` 保留为轻量验证支路。
- “VGGT 初始化”页：增加“VGGT 相机为 BA 提供初始化”。
- “数据接口”页：改为 `VGGT -> SIFT tracks -> BA -> COLMAP text -> Graphdeco`。
- “BA 技术实现”页：从“验证分支”改为“主流程 refinement”，输出是 BA 后相机和 COLMAP 数据。
- “3DGS 优化与展示”页：Graphdeco 输入改为 `BA-refined COLMAP dataset`。
- “最终结果总览”页：主表使用 data1/data2/scene48 的 BA 版结果；scene128 no-BA 只作为 baseline 或补充。
- “Render/GT 对照”页：优先使用 `render_gt_data1_ba_4views.png`、`render_gt_data2_ba_4views.png`、`render_gt_scene48_ba_4views.png`。
- “SIBR 展示”页：默认展示 `data1`、`data2`、`scene48` BA 版；`scene128` 标为 no-BA baseline。
- “BA 对高斯泼溅效果分析”页：使用 data1/data2/scene48 三组同帧数 no-BA vs BA 直接比较；lightweight gsplat 只作为 backup 或补充。

### 新增/更新 PPT 素材

- `ppt_assets/render_gt_data1_ba_4views.png`
- `ppt_assets/render_gt_data2_ba_4views.png`
- `ppt_assets/render_gt_scene48_ba_4views.png`
- `ppt_assets/render_gt_scene48_noba_4views.png`
- `ppt_assets/ba_same_frame_graphdeco_comparison.png`
- `ppt_assets/sibr_windows_transfer.md`
- `official_data1_ba_sibr_package.tar.gz`
- `official_data2_ba_sibr_package.tar.gz`
- `official_scene48_ba_sibr_package.tar.gz`

### BA 对高斯泼溅效果分析三页具体做法

第 14 页：BA 是否改善几何

- 标题：`Bundle Adjustment 降低相机几何误差`
- 目的：先证明 BA 本身有效，避免直接跳到 3DGS 主观图。
- 页面布局：左侧放一张三行表格，右侧放一句结论。
- 表格列：`数据`、`视角数`、`BA 前 RMSE`、`BA 后 RMSE`、`下降幅度`。
- 数值：
  - 数据1：16 views，1.633 -> 1.266 px，下降约 22.5%
  - 数据2：16 views，1.821 -> 1.419 px，下降约 22.0%
  - scene48：48 views，3.215 -> 1.289 px，下降约 59.9%
- 右侧结论写：`BA 直接优化的是 SIFT sparse tracks 的重投影误差；它为 3DGS 提供更一致的相机外参，而不是简单增加点数。`

第 15 页：BA 是否影响 Graphdeco 3DGS

- 标题：`同帧数 no-BA / with-BA 主训练对照`
- 目的：回应“BA 是否对高斯泼溅有效”的评分点，强调这是官方 Graphdeco 30k 主训练结果。
- 页面布局：左侧放 `ppt_assets/ba_same_frame_graphdeco_comparison.png`；右侧放小表格。
- 小表格列：`数据`、`no-BA 高斯数`、`with-BA 高斯数`、`判断`。
- 判断建议：
  - data1：63k -> 40k，BA 后模型更紧凑，需要结合视觉看是否减少漂浮/冗余高斯。
  - data2：63k -> 63k，模型规模基本持平，说明 BA 主要改变几何而不是点数。
  - scene48：939k -> 920k，规模略降但同量级，说明 48 帧 BA 能保持完整场景表达。
- 页脚结论：`高斯数是模型规模指标，不是质量指标；BA 的收益应看同视角 render/GT 和 SIBR 视角切换稳定性。`

第 16 页：视觉对比与结论

- 标题：`BA 对最终渲染的视觉影响`
- 目的：把第 14 页的几何指标和第 15 页的模型规模转成最终效果判断。
- 页面布局：三行，每行一个数据集；每行左侧 no-BA，右侧 with-BA。
- 使用素材：
  - data1：`ppt_assets/render_gt_data1_4views.png` vs `ppt_assets/render_gt_data1_ba_4views.png`
  - data2：`ppt_assets/render_gt_data2_4views.png` vs `ppt_assets/render_gt_data2_ba_4views.png`
  - scene48：`ppt_assets/render_gt_scene48_noba_4views.png` vs `ppt_assets/render_gt_scene48_ba_4views.png`
- 每行只截取 1-2 个最能看出差异的视角，不要整张 contact sheet 全塞进去。
- 右下角放最终结论：`BA 在三组数据上稳定降低重投影误差；进入 3DGS 后不表现为高斯数单调增加，而表现为更一致的相机几何和潜在更稳定的实时浏览效果。`

建议做 20 页正文 + 3-5 页 backup。主线不要按代码文件讲，而是按评分点讲：VGGT 初始化 -> BA 编程实
  现 -> 3DGS 训练与实时展示 -> 改进与消融 -> 结论与未来工作。

  整体结构

   部分                        页数    目标
  ━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1. 项目目标与总流程         2 页    先让老师知道你完整覆盖了 3 个核心任务
  ──────────────────────────  ──────  ──────────────────────────────────────────────
   2. 技术实现讲解             5 页    拿“技术实现讲解与展示”分，讲清楚每一步怎么做
  ──────────────────────────  ──────  ──────────────────────────────────────────────
   3. 最终结果与实时展示       3 页    展示最终可视化质量，证明系统跑通
  ──────────────────────────  ──────  ──────────────────────────────────────────────
   4. BA 对高斯泼溅效果分析    3 页    单独回应 2 分评分点
  ──────────────────────────  ──────  ──────────────────────────────────────────────
   5. VGGT 改进与消融实验      5 页    单独回应改进方法和实验分析
  ──────────────────────────  ──────  ──────────────────────────────────────────────
   6. 总结与未来方向           2 页    收束贡献，拿未来研究方向分

  逐页大纲

    页    主题                   目的                                          建议图表/素材
  ━━━━━  ━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━
     1    标题页：VGGT + BA +    说明任务、数据、最终展示形式                  放 1 张最好的 SIBR/
          3DGS 三维重建                                                        渲染截图作为背景或
                                                                               右侧图
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
     2    作业要求与项目完成     直接对齐评分点，避免老师找不到                表格：VGGT、BA、
          情况                                                                 3DGS、实时展示、消
                                                                               融实验分别对应代码
                                                                               和结果
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
     3    总体 Pipeline          讲清楚数据从输入到 SIBR viewer 的全链路       流程图：输入图像/视
                                                                               频 -> VGGT ->
                                                                               COLMAP text -> 预处
                                                                               理 -> Graphdeco
                                                                               3DGS -> render/SIBR
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
     4    VGGT 初始化：相机与    证明你使用 VGGT 估计相机参数和点云            图：相机 frustum +
          初始点云                                                             初始点云截图；旁边
                                                                               列 pose_enc、
                                                                               world_points、
                                                                               world_points_conf
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
     5    VGGT 到 COLMAP/3DGS    展示工程实现细节，不只是调用模型              图：VGGT 输出字段
          的数据转换                                                           -> COLMAP cameras/
                                                                               images/points3D 映
                                                                               射表
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
     6    BA 编程实现            明确 BA 是自己实现的核心模块                  图：track graph 或
                                                                               公式框：优化相机外
                                                                               参和 3D 点，loss 为
                                                                               重投影误差
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
     7    BA 实现细节：SIFT      讲技术难点和鲁棒性                            图：特征匹配示意、
          tracks +                                                             track 数量统计、
          least_squares                                                        soft_l1 鲁棒损失示
                                                                               意
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
     8    3D Gaussian            覆盖“优化高斯点云”                            图：Gaussian 参数示
          Splatting 实现与训                                                   意：位置、尺度、旋
          练                                                                   转、颜色、opacity；
                                                                               列内部 gsplat 和官
                                                                               方 Graphdeco 两条链
                                                                               路
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
     9    人体数据预处理：       强调你做了有针对性的改进                      图：原图、mask、
          mask、despill、                                                      clean RGBA、训练结
          alpha 背景约束                                                       果前后对比
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
    10    场景数据预处理：       说明场景改进不是只调参，而是几何一致性修正    图：裁剪前后图像；
          letterbox 裁剪与相                                                   标出上下白边；小表
          机内参修正                                                           显示 height/cy 修改
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
    11    最终结果总览           快速给出三个数据集最终成果                    表格：数据1/数据2/
                                                                               场景，视角数、高斯
                                                                               数、PLY大小、输出目
                                                                               录
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
    12    Render/GT 对照展示     展示质量，支撑最终结果                        三组
                                                                               render_vs_gt_contac
                                                                               t_sheet.png，每组截
                                                                               取 3-4 个视角即可
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
    13    实时交互展示方案：     拿“实时交互展示”分                            图：SIBR viewer 截
          SIBR Viewer                                                          图或演示视频截图；
                                                                               旁边放启动命令和
                                                                               run_sibr_viewer.py
                                                                               作用
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
    14    BA 是否改善几何：重    单独回应 BA 分析                              表格/柱状图：新版主
          投影误差分析                                                         流程 BA sparse RMSE，
                                                                               数据1 1.633 ->
                                                                               1.266，数据2
                                                                               1.821 -> 1.419，
                                                                               scene48 3.215 ->
                                                                               1.289
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
    15    BA 是否改善高斯优      直接回答“BA 对高斯泼溅效果”                   图：
          化：同帧数主训练对                                                    ba_same_frame_graphde
          比                                                                   co_comparison.png；
                                                                               表：data1/data2/
                                                                               scene48 的 no-BA vs
                                                                               BA 高斯数和输出目录
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
    16    BA 效果的视觉对比与    把数字转成结论，避免只报指标                  图：data1/data2/
          结论                                                                 scene48 同视角
                                                                               no-BA render/GT vs
                                                                               BA render/GT；结论：
                                                                               BA 降低相机几何误差，
                                                                               高斯数不一定增加，
                                                                               质量看同视角渲染和
                                                                               SIBR 稳定性
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
    17    VGGT 改进 1：          回应“VGGT 改进与消融”                         折线/柱状图：
          confidence                                                           percentile 0/25/50
          filtering                                                            对点数影响；结论：
                                                                               25 是质量与点数折中
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
    18    VGGT 改进 2：输入帧    展示速度/覆盖度权衡                           表格或柱状图：48 帧
          数 sweep                                                             vs 128 帧；建议补：
                                                                               运行时间、最终
                                                                               render 对比
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
    19    输入预处理消融：       强化改进实验分析                              两组前后对比：
          mask/背景约束与场景                                                  masked_soft ->
          裁剪                                                                 masked_clean_bg、
                                                                               uncropped ->
                                                                               cropped；最好用红框
                                                                               标出绿幕漂浮高斯和
                                                                               白边问题
  ─────  ─────────────────────  ────────────────────────────────────────────  ─────────────────────
    20    总结与未来工作         收束贡献，拿未来方向分                        左侧 4 条贡献，右侧
                                                                               未来方向：动态人
                                                                               体、自动选择帧、BA
                                                                               与 3DGS 联合优化、
                                                                               定量指标 PSNR/SSIM/
                                                                               LPIPS

  Backup 建议

   Backup 页    内容                               用途
  ━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          B1    关键命令与复现路径                 老师问“怎么跑”的时候用
  ───────────  ─────────────────────────────────  ──────────────────────────────────
          B2    代码模块对应表                     老师问“哪些是自己写的”时用
  ───────────  ─────────────────────────────────  ──────────────────────────────────
          B3    Graphdeco alpha 背景 loss patch    解释改进细节
  ───────────  ─────────────────────────────────  ──────────────────────────────────
          B4    SIBR viewer 运行环境               解释为什么训练服务器不能实时展示
  ───────────  ─────────────────────────────────  ──────────────────────────────────
          B5    更多 render/GT 图                  回答视觉质量问题

生成的 PPT 素材在：ppt_assets
主要文件：
  - ba_rmse_before_after.png：BA 前后重投影误差
  - gaussian_loss_before_after.png：已有 BA+gsplat 验证里的 Gaussian loss 下降
  - ba_vs_no_ba_gsplat_final_loss.png：no-BA vs BA 对轻量 gsplat final loss 的影响
  - vggt_confidence_filtering_points.png：confidence percentile 对点数的影响
  - vggt_confidence_point_cloud_projection.png：p0/p25/p50 初始点云投影对比
  - scene_frame_count_sweep.png：48 帧 vs 128 帧点数对比
  - human_mask_clean_bg_ablation.png：人体 mask/背景约束前后 render/GT 对比
  - final_human_render_gt_contact_sheets.png：data1/data2 最终 render/GT 对照
  - ba_same_frame_graphdeco_comparison.png：data1/data2/scene48 同帧数 no-BA vs BA 主训练对比
  - render_gt_scene48_noba_4views.png：scene48 no-BA render/GT 对照
  - sibr_windows_transfer.md：拷到 Windows 跑 SIBR 的目录清单和命令

  主 BA 分析现在用 Graphdeco 30k 的同帧数 no-BA/with-BA 对照：
  - outputs/official_data1_graphdeco_30k_masked_clean_bg vs outputs/official_data1_graphdeco_30k_masked_clean_bg_ba
  - outputs/official_data2_graphdeco_30k_masked_clean_bg vs outputs/official_data2_graphdeco_30k_masked_clean_bg_ba
  - outputs/official_scene48_graphdeco_30k_cropped_noba vs outputs/official_scene48_graphdeco_30k_cropped_ba

  旧的 lightweight gsplat no-BA 对照保留为 backup：
  - outputs/verify_data1_no_ba_gsplat_allviews
  - outputs/verify_data2_no_ba_gsplat_allviews
  - outputs/verify_scene_no_ba_gsplat_12views

  verify_scene_no_ba_gsplat_12views 和 verify_scene_no_ba_gsplat_allviews 不再作为主 PPT 的 BA 结论依据，只在老师追问内部 gsplat 验证时使用。

  SIBR 前一步数据已经齐全，不需要再训练。Windows 端需要拷贝的核心目录也写进了 ppt_assets/
  sibr_windows_transfer.md:1。总体大小大概：data1 57M，data2 54M，scene 707M。

[
  \min_{R_i,t_i,X_j} \sum_{(i,j)} \rho(|\pi(K,R_i,t_i,X_j)-u_{ij}|^2)
  ]
