# runs 文件夹说明

更新：2026-11-23。本目录保存训练、评估和可视化产物，不放训练源码或原始数据。容器入口为 `/workspace/srtp-ocean/runs`，持久位置为 `/g3240101584lt/srtp-ocean/runs`，二者访问同一目录。

模型代码和启动参数见 [Code_README.md](../code/Code_README.md)，数据版本与划分见 [Data_README.md](../data/Data_README.md)，历次成绩见 [V2_Trainlog.md](V2_Trainlog.md) 和 [V4_Trainlog.md](V4_Trainlog.md)。本文件只负责解释目录、文件用途及运行状态。

## 文档与指针

| 文件 | 用途 |
| --- | --- |
| `V2_Trainlog.md` | v2模型、基线、消融及v1历史附录 |
| `V4_Trainlog.md` | v4四模型两组种子成绩及测试状态 |
| `train_01_log.md` | 首次训练的性能问题与优化记录 |
| `.current_baseline_suite` | 记录2026-11-20早期对照批次的路径，不代表现在有任务运行 |
| `.current_four_models_200ep` | 记录2026-11-20四模型200轮批次的路径，不指向新增v4批次 |

## 主要训练结果

| 目录 | 保存内容与状态 | 先看什么 |
| --- | --- | --- |
| `three_models_v2_seed20261118_20261123T054818Z/` | v2三模型（Swin、U-Net、ViT），各200轮、seed20261118；各有两段测试报告 | [Results_README.md](three_models_v2_seed20261118_20261123T054818Z/Results_README.md)、各模型 `test_metrics.json` |
| `four_models_200ep_seed20261120_20261120T115114Z/` | v2四模型，各200轮、新种子20261120；子目录为 `residual_cnn`、`unet_baseline`、`vit_baseline`、`swin`，各有两段测试报告 | [Results_README.md](four_models_200ep_seed20261120_20261120T115114Z/Results_README.md)、`suite_summary.json` |
| `baseline_suite_v2_20261120T105835Z/` | v2早期对照；CNN两个种子、U-Net和ViT各一个种子，各250轮；另有气候态与EOF基线。未启动的后续种子已取消 | [Results_README.md](baseline_suite_v2_20261120T105835Z/Results_README.md)、`suite_summary.json` |
| `swin_physics_v2_500ep_gpu_20261118T044643Z/` | v2 Swin 500轮参考实验，含最佳／末轮权重及两段测试 | `config.json`、`metrics.jsonl`、`test_metrics.json` |
| `v4A_four_models_20261121T093236Z/` | v4扩展协议四模型，各200轮，seed20261120；子目录为 `swin`、`cnn`、`unet`、`vit` | 根目录 `halfyear.json`、`halfyear.log`；各模型 `config.json`、`metrics.jsonl` |
| `swin_physics_parallel_data_v2/` | 早期v2 CPU 50轮实验 | `held_out_metrics.json`、`config.json` |
| `swin_physics_v1/`、`swin_physics_parallel_6w_v1/` | v1历史GPU 50轮实验，旧数据划分与采样口径，保留复现 | `config.json`、`metrics.jsonl` |

v2和v4必须按清单及配置区分，不能仅看“200轮”或目录日期判断协议。v2主测试为2024，v4主测试为2022—2024；各自验证期也不同，成绩不直接混排。

v4四个训练日志连续记录第1—200轮，退出码均为0；现有 `halfyear.json` 包含2020H1—2025H1的11个半年、四模型和气候态结果，评估日志所载权重轮次与各模型最低验证分数轮次一致。它不是各模型的 `test_metrics.json`，也没有v2测试报告中的“验证期未见平台”分组。

## 消融、原型与失效批次

| 目录 | 用途与状态 |
| --- | --- |
| `ablation_input_v2_20261121T074157Z/` | 修正版v2输入消融：`no_adt`、`no_wind`、`no_smap`、`no_context`、`no_surface`、`offsets_only`、`oisst_only`、`adt_only`、`wind_only`、`smap_only`，共10个变体；各200轮、退出码0，含逐轮验证，未见独立测试报告 |
| `ablation_20261121T061720Z/A1_lambda*/` | 物理约束系数扫描；同批根目录的 `*.physics.json` 是已有物理违规率诊断 |
| `ablation_20261121T061720Z/B1_*/` | 早期输入消融，既有交接记录标为失效；保留追溯，不用于结论 |
| `ablation_surface_20261121T071141Z/` | 同一失效实现产生的早期批次，部分运行未满200轮 |
| `ablation_input_20261121T072149Z/` | 修复前的失效输入消融批次，保留追溯 |
| `ablation_single_source_20261121T071748Z/` | 早期单源批次的空目录，当前无实验文件 |
| `sparse_anchor_200ep_gpu_20261121T055809Z/` | 稀疏锚点原型，200轮，含权重、测试和 `info_curve.json`；既有记录注明研究方向暂停 |
| `sparse_anchor_smoke_041336/`、`sparse_gpu_smoke_055743/` | 原型短试跑；不作为正式200轮实验 |

失效输入消融的原因在既有记录中为：只修改主进程输入函数，修改没有进入以 `spawn` 启动的DataLoader子进程。现有替代实现为 `code/parallel_training/ablate_site/sitecustomize.py`。本次仅静态读取代码及已有日志，没有重跑注入验证或实验。

## 评估、诊断与图表

| 位置 | 内容 |
| --- | --- |
| `dataset_v2_evaluation/v1_native_validation.json` | v1权重按真实经纬度采样的历史复评 |
| `visualizations/four_models_200ep_seed20261120_with_baselines/` | v2四模型200轮与两统计基线的学习曲线、指标对比及汇总表 |
| `visualizations/baseline_suite_250ep_with_swin500_ref/` | 250轮对照批和500轮Swin参考图；预算、种子次数不一致，保留部分批次标记 |
| `diagnostics/four_models_200ep_seed20261120_diagnose/` | v2四模型的分深度误差、空间误差和典型剖面对比 |
| `diagnostics/four_models_200ep_seed20261120_slices/` | v2四模型的202401、202407温盐切片；具体组合及数据文件见 `slice_manifest.json` |
| `halfyear_eval_20261121T045837Z.json`、`halfyear_eval2_20261121T130657Z.json` | 根目录既有半年评估；后者还含逐月结果。模型、时段以文件内部字段为准 |
| `depth_bands_20261121T132444Z.json` | 既有分深度带评估 |
| `input_coverage_20261121T130748Z.json` | 海表输入逐月掩膜覆盖率 |
| `diag_months_20261121T132108Z_depth.json`、`diag_months_20261121T132108Z_profiles.csv` | 指定月份的深度及剖面诊断 |

`training_ready_v1/` 仅保存早期预处理日志与历史PID，训练数据本体在 `../data/`。`_archive/smoke_runs_20261119T215246/` 保存4个短试跑和1个失败启动目录及恢复清单，未删除。

## 单次运行中的文件

| 文件 | 含义 |
| --- | --- |
| `config.json` | 当次参数、种子、数据位置与训练／验证月份；字段随入口不同，并非所有文件都会记录环境变量或架构名 |
| `metrics.jsonl` | 逐轮训练与验证记录；完整轮次应查看实际epoch序列，不能只看目录名称 |
| `best.pt` | 按该次验证准则选出的检查点，通常用于后续评估 |
| `latest.pt` | 最近保存的训练检查点；能否续训取决于对应训练入口是否支持恢复 |
| `test_metrics.json`、`held_out_metrics.json` | 已有测试或综合评价报告，二者结构不同，应先读字段及月份 |
| `train.log`、`<model>.log`、`<model>.driver.log` | 单次或外层调度日志，实际位置随启动脚本不同 |
| `<model>.exitcode` | 调度脚本保存的退出码；0表示脚本成功退出，不单独证明科学口径完整 |

批次根目录还可能有 `suite_config.json`、`suite_summary.json`、`source_sha256.json` 和旧成果校验清单。旧批次的 `*.cancelled_controller.exitcode` 记录后续种子控制器被取消，不等同于已完成训练失败。历史 `pid` 和隐藏路径指针均不能代替当前进程检查。

## 查找与留存约定

先从本文件定位目录，再读该次配置、逐轮指标及已有报告；数值汇总按协议查 `V2_Trainlog.md` 或 `V4_Trainlog.md`。不在本目录放模型源码，不按测试分数覆盖或替换旧实验。新训练使用独立目录，已有权重、日志和失效批次保留来源记录。

本次仅检查文本、JSON、CSV及文件存在性，未重新加载权重、计算预测或重做数据验收；只修改说明文档并恢复项目入口软链接。

<!-- V4_SEED_20261122 -->
## v4 新种子训练（2026-11-22，已完成）

`v4A_four_models_seed20261122_20261122T053535Z/`：已完成的v4四模型seed20261122实验，各200轮，独立测试未运行。目录根保留模型日志、退出码、[结果说明](v4A_four_models_seed20261122_20261122T053535Z/Results_README.md)及training_summary.json；各模型子目录含config.json、metrics.jsonl、best.pt与latest.pt。launch_provenance.json和previous_weights_sha256.json分别记录来源与99个旧权重基准。
<!-- END_V4_SEED_20261122 -->

## v2 seed=20261118 三模型训练与测试（2026-11-23，已完成）

`three_models_v2_seed20261118_20261123T054818Z/`：在 v2 协议下并行完成 `swin`、`unet_baseline`、`vit_baseline` 各 200 轮训练与两段测试（`test_locked` 2024、`forward_test` 2025H1），退出码均 `0`。目录根保留 [Results_README.md](three_models_v2_seed20261118_20261123T054818Z/Results_README.md) 及各模型日志、退出码；各模型子目录含 `config.json`、`metrics.jsonl`、`best.pt`、`latest.pt` 与 `test_metrics.json`。
