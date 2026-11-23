# 训练结果对比可视化

独立读取 `config.json`、`metrics.jsonl` 和 `test_metrics.json`，生成可导出的科学图和CSV汇总。只读输入，输出到新文件夹；不加载权重、NetCDF、训练代码或CUDA，不启动训练与推理。

## 图表与表格

- 每个模型／配置的学习曲线：训练总目标、验证分数、CT及SA验证RMSE，标记验证选出的最佳轮。
- 测试RMSE、R²、偏差：2024年和2025年上半年分栏，全部E与验证期未见E分别作图。
- 多种子：均值、样本标准差、每个种子的原始点，标出完成数，例如 `n=1/3`。
- 完成轮数与每轮耗时：耗时使用日志中的训练＋验证时间，去掉前5轮后取中位数；不包含权重写入和最终测试，并发时也不能解释为独立硬件基准。
- `run_inventory.csv`、`test_scores_per_run.csv`、`test_summary.csv` 和带源文件SHA256的 `comparison.json`。

图中文字为英文，默认输出PNG和PDF，也可选择SVG。CT单位为℃，SA为g/kg。误差棒表示种子间样本标准差，不是置信区间；单次统计基线不伪造标准差。

多种子指标的平均值描述各次训练的表现，不等于把预测做集成后的表现。

## 使用

Python 3.11及以上；依赖见 `requirements.txt`。可在安装了NumPy、Matplotlib的本地CPU环境运行，服务器上也能运行。本工具无需导入项目的训练依赖。

训练全部完成后：

```bash
bash /workspace/srtp-ocean/code/training_visualization/run_visualization.sh \
  --runs-root /workspace/srtp-ocean/runs/baseline_suite_v2_20261120T105835Z \
  --output-dir /workspace/srtp-ocean/runs/visualizations/baselines_final_20261120
```

训练中查看快照，加 `--allow-partial` 并使用新的输出目录，图上会注明 `PARTIAL SNAPSHOT`。默认要求CNN、U-Net、ViT的三个种子都完成两段测试；缺失结果不会填零。自定义种子用 `--expected-seeds 42 43 44`。

可额外加入旧Swin作为单次参考：

```bash
bash /workspace/srtp-ocean/code/training_visualization/run_visualization.sh \
  --runs-root /workspace/srtp-ocean/runs/baseline_suite_v2_20261120T105835Z \
  --reference-run /workspace/srtp-ocean/runs/swin_physics_v2_500ep_gpu_20261118T044643Z \
  --output-dir /workspace/srtp-ocean/runs/visualizations/baselines_with_swin_20261120 \
  --formats png pdf svg
```

旧Swin为500轮，新模型为250轮，图例和报告会注明预算不同，只作描述性参考。验证曲线中的星号按验证分数选择，不按测试结果挑选。已有v1或缺少原生经纬度采样记录的运行不接受直接混入。

本地Windows可在 `服务器留存/code` 下执行：

```powershell
python -m training_visualization --runs-root '你的日志快照目录' --output-dir '新的绘图输出目录' --allow-partial
```

## 比较口径与限制

- 不同数据清单SHA256不能混合；同一架构不同超参数会分组，不会误当成不同种子合并。
- 同一测试时段和浮标组必须具有相同月份、有效点数、剖面数及平台数。
- 上述检查核对报告中的月份和计数；汇总JSON不足以逐条核验观测身份。
- 每份测试报告必须使用日志中的验证最佳轮。损坏的中间日志记录会报错；正在追加的末尾不完整JSON行可以跳过并记录。
- R²是有效剖面与深度合并后的总体统计，不能据此推断各深度表现。Argo E可能进入CORA上游分析，仍保留原有观测重合限制。
- 汇总JSON没有逐点观测和预测，不足以生成空间图、预测散点图或分深度误差；本工具不重新运行模型来补造这些内容。
- 不覆盖既有图表；每次使用新的 `--output-dir`，便于追溯快照。

## 代码检查

公共API有类型注解和文档字符串，120字符行宽，绝对导入。代码格式参照 `python-code-style`。

```bash
ruff check --config training_visualization/pyproject.toml training_visualization
ruff format --check --config training_visualization/pyproject.toml training_visualization
mypy --config-file training_visualization/pyproject.toml training_visualization
python -m pytest training_visualization/tests -q
```
