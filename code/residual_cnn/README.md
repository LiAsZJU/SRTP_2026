# Residual CNN baseline

The model takes `[B, 276, 81, 321]` and returns normalized CT/SA anomalies with
shape `[B, 2, 60, 81, 321]`. A 1x1 projection produces 48 features, followed by
four residual blocks with dilation 1/2/4/1 and a 120-channel regression head.
GroupNorm works with the existing one-month-per-batch setup.

Training reuses the v2 CORA grid loss, native-coordinate Argo Q loss and vertical
stability penalty. Validation E selects `best.pt`; no test data choose weights.
Defaults: 200 epochs, patience 500, AdamW 1e-4,
6 training loader processes, 2 validation loader processes, 1 main CPU thread,
CUDA AMP and 8 GiB device cache. This is not an equal-parameter comparison;
the actual parameter count is saved in each run's configuration.

```bash
# One seed; run only after opening a GPU environment.
bash /workspace/srtp-ocean/code/residual_cnn/run_gpu.sh

# Three fresh runs, sequentially, followed by mean/sample-std aggregation.
bash /workspace/srtp-ocean/code/residual_cnn/run_seeds.sh 20261117 20261118 20261119
```

Use `SEED`, `EPOCHS`, `PATIENCE`, `WORKERS`, `VALIDATION_WORKERS`, `CACHE_GIB`,
`DATASET_DIR` and `RUN_DIR` environment variables for a single run. Model options
can follow the script, for example `--width 48 --blocks 4 --no-amp`.
For repeated runs, set `GROUP_DIR` for the parent directory; use the same
hyperparameters for all seeds. Python, NumPy and both PyTorch CPU/CUDA RNGs are
seeded. CUDA grid_sample backward can still be nondeterministic; exact bitwise
reproducibility across runs or environments is not promised.

All 80 training and 24 validation months are used, with no data shortcuts. Each
successful run evaluates its best checkpoint on 2024 and 2025-01/06, reporting
all E and validation-unseen E. See `../climatology_baseline/README.md` for shared
dependencies and artifacts. Output prefix: `runs/residual_cnn_v2_`.

Only static checks were performed; no CPU/GPU training, model forward pass or
dataset evaluation was run during delivery.

2026-11-20 execution update: requested GPU runs and both test periods completed. Actual seed counts are CNN=2, U-Net=1, ViT=1; the deterministic references ran once each. Remaining repeats were cancelled at the user's request. See runs/baseline_suite_v2_20261120T105835Z/Results_README.md. No further training was started.

2026-11-20: Future runs default to at most 200 epochs. Existing early stopping and completed 250-epoch results are retained. No training launched.
