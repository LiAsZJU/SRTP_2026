# data 数据目录说明

更新：2026-11-23。本轮核对目录、v2／v4清单、配置和既有报告；未重新处理数据或进行全期科学验收。

容器路径：`/workspace/srtp-ocean/data`
对应平台个人存储：`/g3240101584lt/srtp-ocean/data`

Argo的22个归档包、2份上传校验清单及核验脚本已移入 `zip_s/argo_zip/`，服务器文件数核对为25。

## 当前训练清单

原主协议为v2，v4为同学新增的扩展实验；两版同时保留，不互相覆盖。`training_ready_v4/` 保存独立清单、训练统计及审计，详见文末。

| 入口 | 训练／验证／主测试／补充测试月数 | SMAP口径 |
| --- | --- | --- |
| `training_ready_v2/training_manifest.json` | 80／24／12／6 | 原v2海表输入协议 |
| `training_ready_v4/training_manifest.json` | 179／24／36／6 | 训练和评估启动环境通过 `ABLATE_MODE=no_smap` 与 `ablate_site` 关闭 |

v4清单仍保留海表数据引用，“不使用SMAP”由加载时的代码与环境控制，不能仅凭清单路径判断。v4的 `halfyear.json` 是批次级半年E组评价，当前没有v2报告的验证未见平台分组。两版划分和输入不同，成绩分开报告。

## 目录与文件

| 名称 | 内容与用途 | 当前核验状态 |
| --- | --- | --- |
| `grid_template.nc` | 共同网格模板：81个纬度点、321个经度点、60个深度层；提供坐标、边界和静态域 | 网格读取通过；服务器哈希尚未复核 |
| `oisst/` | 已处理的OISST海表温度月场，主变量为 `sst` | 200411—202506，共248个月，已解包并核对文件数；服务器包哈希待复核 |
| `adt/` | 已处理的绝对动力地形月场，主变量为 `adt` | 200411—202506，共248个月，已解包并核对文件数；服务器包哈希待复核 |
| `argo/` | Argo温盐剖面预处理结果、完整集合索引及相关来源记录 | 全量解包和哈希校验通过，详见下文 |
| `zip_s/argo_zip/` | Argo初始上传材料，共25个文件；与解包后的 `argo/` 分开存放 | 迁移完成，文件数25 |
| `cora/` | 同学上传的原始CORA fld温盐文件 | 已见；201801温度属性抽查通过 |
| `training_ready_v1/` | CORA SA/CT、Argo C网格与Q/E剖面、训练背景/归一化及统一清单 | 每类月度结果246份；三个月联合读取通过；已有passed审计不覆盖全部科学问题 |
| `era5/` | ERA5 月场，训练读取 `u10`、`v10` | 201801 纬度抽查及三个月的联合读取通过，全期交付未重审 |
| `smap/` | SMAP 海表盐度月场，训练读取 `sss_smap` | 201801 纬度抽查及三个月的联合读取通过，全期交付未重审 |
| `era5_smap_server_data_v1/` | ERA5/SMAP交接包解包目录 | 目录已可见；与顶层 `era5/`、`smap/` 的对应关系尚待核对 |
| `zip_s/oisst_adt_200411_202506_v2.tar` | OISST/ADT传输归档，已解包至对应目录 | 科学数据读取使用解包后的文件 |
| `zip_s/argo_zip/argo_2005_v1.tar`—`zip_s/argo_zip/argo_2025_v1.tar` | Argo的21个年度传输归档 | 全部服务器哈希通过，已解包 |
| `zip_s/argo_zip/argo_metadata_v1.tar` | Argo集合索引、批次配置及来源记录等元数据归档 | 服务器哈希通过，已解包 |
| `zip_s/era5_smap_server_data_v1.tar.gz` | ERA5/SMAP交接归档 | 文件已可见，尚未核验 |
| `zip_s/SHA256SUMS` | OISST/ADT归档校验清单 | 早期服务器副本有CRLF换行问题；修复后的校验结果尚未返回 |
| `zip_s/argo_zip/ARGO_SHA256SUMS` | 全部22个Argo归档的校验清单 | 服务器全部显示 `OK`；再次校验时在 `zip_s/argo_zip/` 内执行 |
| `zip_s/argo_zip/FIRST_UPLOAD_SHA256SUMS` | Argo首批试传文件的校验清单 | 用于元数据包、2005年包及核验脚本的首批检查 |
| `surface_upload_manifest_v2.json` | OISST/ADT交付文件及校验信息清单 | 用于追溯交付内容 |
| `zip_s/argo_zip/verify_argo_upload.py` | Argo解包结果的结构、数量、相对引用及可选逐分片哈希核验工具 | 全量 `--hash` 检查通过；默认数据入口不受脚本移动影响 |

CORA dat 来源审计已在 v2 中记录。TAO/TRITON 评价待开展。

## OISST与ADT读取位置

```text
oisst/months/YYYYMM/oisst_YYYYMM.nc
adt/months/YYYYMM/adt_YYYYMM.nc
```

`YYYYMM` 表示六位年月，例如 `201801`。两类月场均已映射到共同的81×321水平网格，通常为 `[time=1, latitude=81, longitude=321]`。

处理顺序为先计算源网格月均，再进行面积重映射；文件同时保存有效天数、面积支持及缺测信息。各月的 `completed.json`、`source_inventory.json`、`validation.json` 分别记录完成状态、输入来源和处理检查结果。

## Argo读取位置

统一入口：`argo/outputs/argo_complete_v2/collection.json`。

```text
argo/
  argo_collection.py
  outputs/
    argo_complete_v2/collection.json
    full_200501_202506_v1/months/YYYYMM/profiles_*.nc
    legacy_5910_patch_v1/months/YYYYMM/profiles_*.nc
  code_versions/
  repair_runs/
  reviews/
```

| 项目 | 已验证结果 |
| --- | --- |
| 时间范围 | 200501—202506，246个月 |
| 主批次 | 466860条剖面 |
| 旧格式修复补充批次 | 5910条剖面 |
| 合计 | 472770条剖面、4044个NetCDF分片 |
| 服务器核验 | `status=upload_verified`、`scope=full_collection`、`all_shard_hashes_checked=true` |

按总清单的 `months[YYYYMM]` 读取该月列出的全部分片，包含主批次与补充批次；只读主批次会漏掉5910条补充剖面。清单使用相对路径，应保持 `argo/` 内的层级关系。

Argo保留真实经纬度、观测时间和浮标身份，不是完整的规则水平场。各分片的 `SA`（绝对盐度，g/kg）、`CT`（保守温度，℃）为 `[profile, depth]`，深度共60层：5—100 m每5 m一层，110—500 m每10 m一层。另保留有效掩膜、原观测层、QC及来源信息。

原观测层表的索引在各分片内独立；跨分片合并时需处理索引偏移，不能直接拼接所有字段。原始预处理集合保留不变；同学已在 `training_ready_v1/` 派生产物中划分C/Q/E及月份用途。

## 训练准备版本

`training_ready_v1/training_manifest.json` 是训练读取入口。它引用 `cora/`、`argo_grid/` 和 `argo_profiles/` 各246个月的派生产物，以及 `statistics/` 下的背景和归一化统计。这里的三个子目录均位于 `training_ready_v1/` 内。训练背景及归一化只用201504—202011拟合。

按目标格点中心范围筛选后，训练准备版包含 C=187229、Q=188112、E=88235，共463576条剖面。C 是输入，Q 是训练监督，E 是按平台留出的评价数据。原始 Argo 集合仍通过上文的 `collection.json` 读取。

| 用途 | 目标月份 | 月数 |
| --- | --- | ---: |
| 早期归档，当前未进入预训练 | 200501–201503 | 123 |
| 训练 | 201504–202011 | 68 |
| 隔离月 | 202012 | 1 |
| 验证 | 202101–202212 | 24 |
| 主测试 | 202301–202412 | 24 |
| 补充测试 | 202501–202506 | 6 |

`training_ready_v1/audits/training_readiness.json` 保存已有检查结果。另已抽查201504、201801、202101的联合读取。Argo 剖面采样的纬度偏差及复评范围见服务器 `code/Code_README.md`。

## 状态说明

解包目录用于读取科学数据，归档包用于传输留存；Argo上传材料统一归入 `zip_s/argo_zip/`。归档与解包结果包含重复内容，不应重复计入样本。

Argo完整性核验通过，表示文件、清单及校验值一致，不表示训练验收完成。OISST/ADT的正式支持度阈值，以及ERA5/SMAP的交接口径与实际文件，仍需后续核对。

## 训练准备 v2

`training_ready_v2/training_manifest.json` 使用 80 个训练月（201504–202111）、202112 隔离月、24 个验证月（202201–202312）、12 个主测试月（202401–202412）和 6 个补充测试月（202501–202506）。200501–201503 继续作为历史档案。

v2 复用 v1 的 CORA 和 Argo 月文件，并另存按 80 个训练月重新拟合的 `statistics/`。202312、202412 的 SMAP 已各用 27 个合规 RSS 原始窗口重建，两个重建月分别剔除 4 个跨界窗口。来源子集存放在 `sources/`。各月时窗、重建来源和 CORA–Argo 匹配记录见 `audits/`。

`evaluation_platforms.json` 固定评估浮标分组。2024 年主测试覆盖 144 个 E 浮标，其中 26 个未在 2022–2023 年验证中出现。CORA 观测清单仍含部分 E 观测，两组结果均按留出 Argo 口径解读。训练和验证方法见 [数据集 v2 代码说明](../code/dataset_v2/README.md)。

A800 六进程试跑直接读取本节 v2 清单，沿用已发布的月份划分、训练背景和归一化统计。

训练结束后自动测试 2024-01 至 2024-12（12 个月）和 2025-01 至 2025-06（6 个月）。两个时段分别统计全部 Argo E 与验证期未见浮标的成绩。

2026-11-19 核验 500 轮实验仍使用 v2 的 80／24／12 划分，2025 年上半年为 6 个月补充测试。运行配置、测试报告及现有清单 SHA-256 均为 `b7e9924c30358a7c2f6e3c772a14366515b1f3b321e992e6b3b77604c48b4555`。本次核对覆盖配置和报告，未逐个重算科学数据文件哈希。

各版数据对应的训练成绩见 [V2_Trainlog.md](../runs/V2_Trainlog.md) 和 [V4_Trainlog.md](../runs/V4_Trainlog.md)。

2026-11-19：四套对照入口复用现有v2清单和仅训练月拟合的统计，不生成新数据。EOF拟合只用双变量60层均有效的训练列，逐月保存完整列数量；浅水及不完整列不参与拟合。该统计基线与CNN的损失不同。本轮未运行任何数据拟合或评估。

## 上传归档位置

2026-11-19：传输归档集中到 `zip_s/`，解包数据和训练清单路径保持原位。

```text
zip_s/
  oisst_adt_200411_202506_v2.tar
  era5_smap_server_data_v1.tar.gz
  SHA256SUMS
  argo_zip/  # 25 个 Argo 上传材料
```

校验 OISST/ADT 包时在 `zip_s/` 执行；Argo 包校验在 `zip_s/argo_zip/` 执行。

模型实现及训练入口统一见 [Code_README.md](../code/Code_README.md)。

2026-11-19：ViT复用现有v2清单、输入和非均匀经纬度采样；未重制数据。训练轮次默认250不改变数据划分。

<!-- BASELINE_SUITE_20261120 -->
2026-11-20：五模型复用v2清单，实际训练读取80个训练月、验证24个月，并测试2024年及2025年上半年。EOF拟合1,887,520个完整训练列。数据未重制；各次配置／测试报告的清单哈希一致。后续重复按用户要求未启动。
<!-- END_BASELINE_SUITE_20261120 -->
2026-11-20：Swin种子设置仅影响后续模型初始化及加载器随机状态；v2数据、角色划分、月份划分不变。

2026-11-20：残差CNN、U-Net、ViT、Swin后续训练默认上限统一为200轮，保留原提前停止设置；历史配置、成绩和v2数据划分不变，本次未启动训练。

2026-11-20：CNN、U-Net、ViT、Swin新种子20261120各200轮及两段测试已完成；旧成果69文件（含28权重）校验值不变。结果目录 runs/four_models_200ep_seed20261120_20261120T115114Z/，未安排后续训练。

2026-11-20：训练简报已按最佳实验、四模型分类及时间顺序重排，收录19条记录；仅整理文档。

## 训练准备 v4（方案 A，2026-11-21 构建）

`training_ready_v4/training_manifest.json` 是 v4 协议的训练读取入口。构建脚本：`code/dataset_v2/build_dataset_v4.py`（`fit` 重拟合统计量，`publish` 生成清单与审计）。

| 用途 | 目标月份 | 月数 | E 组剖面 / 平台 |
| --- | --- | ---: | --- |
| 训练 | 2005-01 至 2019-11 | 179 | — |
| 隔离月 | 2019-12 | 1 | — |
| 验证 | 2020-01 至 2021-12 | 24 | 9,021 / 165 |
| 主测试 | 2022-01 至 2024-12 | 36 | 17,973 / 180 |
| 补充测试 | 2025-01 至 2025-06 | 6 | 3,224 / 145 |

要点：

- **统计量只拟合一次**：`statistics/training_background.nc` 与 `statistics/normalization.json` 的 `fitted_months` 严格等于 179 个训练月，publish 时校验；
- **不使用 SMAP**：v4 全流程（训练/验证/测试）统一无 SMAP，2015 年前本为零掩膜占位，2015–2019 用 `ABLATE_MODE=no_smap` 在加载时置零，保证年代间输入一致；因此 SMAP 窗口界只记录、不强制截断（`audits/time_boundaries.json` 中 `smap_cutoff_enforced=false`），OISST/ADT/ERA5 仍强制审计；
- **CORA/Argo 与 v1 共用**：v4 复用 v1 的 CORA 与 Argo 月文件，不复制大文件；只新增统计量与清单；
- **SMAP 边界文件**：202312/202412 的 SMAP 路径复用 `training_ready_v2/smap_boundary/`；
- **评估浮标分组**：`evaluation_platforms.json` 固定验证/测试/前向的 E 平台集合；
- **v2 保留不动**：v2（80/24/12/6）仍为历史实验与 500 轮 Swin 的口径，两套协议结果不可直接比较。

构建命令：

```bash
cd /workspace/srtp-ocean/code
export PYTHONPATH=$PWD/python_pkgs:$PWD
python dataset_v2/build_dataset_v4.py fit      # 背景 + 归一化（CPU）
python dataset_v2/build_dataset_v4.py publish  # 清单 + 审计
```

v4 四模型训练与评估：

```bash
bash code/parallel_training/run_v4_four_models.sh          # 4 模型并行，200 轮
# 或单模型：
PYTHONPATH=$PWD/code/parallel_training/ablate_site:$PWD/code/python_pkgs:$PWD/code:$PWD/code/parallel_training \
ABLATE_MODE=no_smap python code/parallel_training/train_arch.py --arch swin \
  --dataset-dir /workspace/srtp-ocean/data/training_ready_v4 --run-dir /workspace/srtp-ocean/runs/<新目录> \
  --device cuda --workers 2 --validation-workers 1 --amp --cache-gib 14 \
  --epochs 200 --patience 201 --seed 20261120 --dim 48 --blocks 4 --heads 4 --window 8
```

半年节点评估（2020H1–2025H1）：

```bash
PYTHONPATH=$PWD/code/parallel_training/ablate_site:$PWD/code/python_pkgs:$PWD/code:$PWD/code/parallel_training \
ABLATE_MODE=no_smap python code/dataset_v2/halfyear_eval_v4.py \
  --suite /workspace/srtp-ocean/runs/v4A_four_models_20261121T093236Z \
  --output /workspace/srtp-ocean/runs/v4A_four_models_20261121T093236Z/halfyear.json
```

2026-11-21：完成 v4 数据集构建（fit 与 publish 均通过：训练 179 / 隔离 1 / 验证 24 / 测试 36 / 前向 6，唯一文件路径 1,732 个）。本次只新增 v4 数据与统计量，未改动 v1/v2 任何文件。

2026-11-22只读复核：v2清单SHA256为 `b7e9924c30358a7c2f6e3c772a14366515b1f3b321e992e6b3b77604c48b4555`；v4为 `80fb336075a251c3d5ef1cba0de558d7c11791ac124b42a03a44d13184a996cc`。本轮检查清单及归一化统计中的月份记录，未重算背景NetCDF或逐一核验所有输入文件。项目软链接已恢复，数据本体未变。

<!-- V4_SEED_20261122 -->
## v4 新种子训练（2026-11-22，已完成）

本轮v4新种子20261122训练已完成，复用现有training_ready_v4，未修改数据。训练179月、验证24月，运行时no_smap；manifest SHA256仍为80fb336075a251c3d5ef1cba0de558d7c11791ac124b42a03a44d13184a996cc。本轮尚未对2022—2024主测试或2025H1补充测试评分。
<!-- END_V4_SEED_20261122 -->

## v2 seed=20261118 三模型复用记录（2026-11-23）

Swin、U-Net、ViT 以 `seed=20261118` 复用 `training_ready_v2/training_manifest.json`（80 训练月／24 验证月／12 主测试月／6 补充测试月，manifest SHA-256 `b7e9924c30358a7c2f6e3c772a14366515b1f3b321e992e6b3b77604c48b4555`）完成 200 轮训练及两段测试，保留完整 SMAP 海表输入，未修改任何数据文件或统计量。
