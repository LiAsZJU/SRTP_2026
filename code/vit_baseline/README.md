# Seeded ViT regression baseline

A compact Vision Transformer maps `[B, 276, 81, 321]` inputs to normalized
CT/SA anomalies shaped `[B, 2, 60, 81, 321]`. The 3x3 patch projection creates
2,889 tokens. Each of four independently initialized encoder blocks applies
global attention with four heads and embedding dimension 96. A learned position
embedding and transposed-convolution regression head restore the native grid.
There is no classification token, softmax output or pretrained weight download.

The model uses the same v2 reader, native-coordinate Argo operator, grid/Q/physics
objective, validation selection and two test splits as CNN and U-Net. Shared
infrastructure lives in `../climatology_baseline/`. Retain all dependencies.

## Future GPU execution

```bash
# One run, default seed 20261117 and maximum 200 epochs.
bash /workspace/srtp-ocean/code/vit_baseline/run_gpu.sh

# Choose one seed explicitly.
SEED=42 bash /workspace/srtp-ocean/code/vit_baseline/run_gpu.sh

# Three fresh sequential runs; defaults are 20261117, 20261118, 20261119.
bash /workspace/srtp-ocean/code/vit_baseline/run_seeds.sh

# Custom repeat seeds.
bash /workspace/srtp-ocean/code/vit_baseline/run_seeds.sh 42 43 44
```

Direct Python training also accepts `--seed`; the shell launcher uses `SEED` so
its run directory, training and evaluation agree. Python, NumPy and PyTorch RNGs
are seeded before model construction. CUDA grid_sample backward can remain
nondeterministic; exact bitwise repeatability is not guaranteed.

Defaults: at most 200 epochs, patience 500, learning rate 1e-4, CUDA AMP, six
training loader workers, two validation workers, one main CPU thread, 8 GiB
device cache. `EPOCHS` can explicitly override the default. Model flags include
`--embedding-dim 96 --blocks 4 --heads 4 --dropout 0.0` after the single-run script.

The launcher rejects CPU fallback and evaluates `best.pt` on 2024 and 2025-01/06
only after training. Reports cover all E and validation-unseen E; repeat runs
produce a mean/sample-standard-deviation summary. Run output prefix is
`runs/vit_baseline_v2_`. The historical CORA/Argo overlap caveat still applies.

Only static checks are authorized for this delivery. No model forward pass,
fitting, training or dataset evaluation is performed. CUDA memory use and
numerical behavior remain untested; global attention can cost more than Swin.

2026-11-20 execution update: requested GPU runs and both test periods completed. Actual seed counts are CNN=2, U-Net=1, ViT=1; the deterministic references ran once each. Remaining repeats were cancelled at the user's request. See runs/baseline_suite_v2_20261120T105835Z/Results_README.md. No further training was started.

2026-11-20: Future runs default to at most 200 epochs. Existing early stopping and completed 250-epoch results are retained. No training launched.
