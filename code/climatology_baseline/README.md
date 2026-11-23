# Monthly climatology reference

This reference predicts the calendar-month CT/SA background already fitted on
the 80 v2 training months (2015-04 through 2021-11). It ignores current sea-surface
and Argo C measurements. There is no new optimizer or iterative training.

The model returns `-anomaly_mean / anomaly_std` in normalized units, so converting
back to physical units recovers the monthly background, without adding a spurious
anomaly mean. Its recorded epoch is 0. Do not interpret it as a trained CNN.

## Future CUDA execution

```bash
bash /workspace/srtp-ocean/code/climatology_baseline/run_gpu.sh
```

No CPU fitting, model forward pass or dataset evaluation was run during delivery.
The launcher refuses a machine without CUDA. On a later GPU run it packages the
reference, validates against 24 validation months, then tests 2024 and 2025-01/06.

`SEED=20261117` is the default. A repeated-seed entry is available:

```bash
bash /workspace/srtp-ocean/code/climatology_baseline/run_seeds.sh 20261117 20261118 20261119
```

There is no random model initialization here; repeat runs are reproducibility
controls, not evidence of stochastic training stability. Running once is normally
sufficient. Seed summaries report sample standard deviation, with denominator n-1.

## Shared implementation

- `runtime.py`: CUDA guard, all random seeds, v2 provenance and neural training.
- `evaluate.py`: architecture-aware evaluation with the existing native-coordinate
  Argo operator and pooled CT/SA RMSE, bias and R-squared.
- `launch_model.sh`: one fresh run and both test splits, in the foreground.
- `launch_seeds.sh`: sequential fresh runs with distinct seeds; no GPU contention.
- `summarize_seeds.py`: aggregate completed JSON reports, without loading models.
- `pyproject.toml`: Ruff and strict mypy settings for all five baseline packages.

All five new directories (including vit_baseline) and the existing v2/parallel-training modules are needed.
Use their launchers rather than executing individual files from arbitrary paths.
The existing Swin trainer, checkpoints and dataset are unchanged.

Fresh outputs go to `runs/climatology_baseline_v2_seed<seed>_<UTC>_<PID>/`.
Artifacts include `config.json`, `metrics.jsonl`, `best.pt`, `latest.pt`,
`test_metrics.json` and a sibling `.log`. Repeated runs additionally create
`seed_summary.json` in their group directory. Existing outputs are never reused.

The common evaluator uses validation E profiles for selection, and reports both
all E and platforms unseen in validation on each test split. These are not fully
independent of upstream CORA observations; retain the existing provenance caveat.

## Static checks

Only source formatting, linting, type checking, Python compilation and `bash -n`
are authorized for this CPU-only delivery. CUDA behavior remains untested.
Python 3.11 matches the inspected server environment. Public APIs have annotations
and Google-style docstrings. Strict mypy checks the new code; legacy data,
loss and evaluator modules are imported as external interfaces, not newly typed.

```bash
cd /workspace/srtp-ocean/code
ruff check --config climatology_baseline/pyproject.toml climatology_baseline eof_ridge residual_cnn unet_baseline vit_baseline
ruff format --check --config climatology_baseline/pyproject.toml climatology_baseline eof_ridge residual_cnn unet_baseline vit_baseline
```

2026-11-20 execution update: requested GPU runs and both test periods completed. Actual seed counts are CNN=2, U-Net=1, ViT=1; the deterministic references ran once each. Remaining repeats were cancelled at the user's request. See runs/baseline_suite_v2_20261120T105835Z/Results_README.md. No further training was started.
