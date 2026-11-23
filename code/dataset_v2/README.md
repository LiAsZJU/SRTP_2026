# 数据集 v2

| 用途 | 月份 | 月数 |
|---|---|---:|
| 历史档案 | 2005.01–2015.03 | 123 |
| 训练 | 2015.04–2021.11 | 80 |
| 隔离 | 2021.12 | 1 |
| 验证 | 2022.01–2023.12 | 24 |
| 主测试 | 2024.01–2024.12 | 12 |
| 补充测试 | 2025.01–2025.06 | 6 |

训练读取 `/workspace/srtp-ocean/data/training_ready_v2/training_manifest.json`。月背景场和标准化只用训练块重新拟合。SMAP 按产品窗口的真实结束时间筛选再求月代表场，2023 和 2024 年的 12 月各使用 27 个窗口。OISST、ADT、ERA5 检查文件记录的月末边界，其中 OISST 和 ADT 的上游同化观测范围不由月文件给出。历史海表输入允许跨名义年份。窗口来源见 [RSS V6.0 数据](https://doi.org/10.5067/SMP60-3SPCS)。

## 训练与评估

以下命令创建一个独立训练目录。在 CPU 容器可用 `--device cpu --no-amp --cpu-threads 8 --workers 4` 代替 GPU 设置。

```bash
cd /workspace/srtp-ocean
bash code/parallel_training/run_parallel.sh \
  --dataset-dir data/training_ready_v2 --run-dir runs/swin_physics_v2_next \
  --device cuda --workers 6 --validation-workers 2 --cache-gib 8

export PYTHONPATH="$PWD/code/python_pkgs:$PWD/code"
PYTHONDONTWRITEBYTECODE=1 /opt/conda/bin/python code/dataset_v2/evaluate_checkpoint.py \
  --dataset-dir data/training_ready_v2 --checkpoint runs/swin_physics_v2_next/best.pt \
  --output runs/swin_physics_v2_next/held_out_metrics.json \
  --splits validation test_locked forward_test --device cpu
```

`grid_sampler.py` 用实际经纬度轴做双线性插值。Q 监督与 E 验证使用同一个算子。程序以清单哈希校验模型与数据集的对应关系。

主测试同时报告所有 E 浮标，以及从未出现在验证期 E 组的浮标。RMSE 和 R² 汇总全部有效深度点。CORA 观测清单存在同浮标、同时间和同位置的匹配，因此浮标留出应结合 `data/training_ready_v2/audits/cora_e_overlap.json` 解读。匹配审计覆盖同标称月清单，温盐两项分别计数。数据出处见 [Copernicus CORA 手册](https://documentation.marine.copernicus.eu/PUM/CMEMS-INS-PUM-013-001.pdf)。

## 处理脚本

- `build_dataset.py` 的 `fit` 拟合统计量，`publish` 检验清单、构建评估组并发布数据入口。
- `fetch_smap_boundary.py` 从 RSS 官网抓取合规窗口的源网格子集。若服务器无法访问站点，可在有外网的机器执行。源子集 ZIP 已上传到 v2 的 `sources/`。
- `rebuild_smap_boundary.py` 读取解包的窗口子集，重建月场并记录原始窗口来源。
- `audit_cora_overlap.py` 按平台、UTC 时间和观测位置检查 CORA 清单与 E 剖面的重合。
