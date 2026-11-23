# U-Net regression baseline

The model takes `[B, 276, 81, 321]` and returns normalized CT/SA anomalies with
shape `[B, 2, 60, 81, 321]`. Encoder widths are 24/48/96 with a 192-channel
bottleneck. Three average-pooling stages, skip connections and bilinear resizing
preserve the odd native 81x321 output grid. The head is continuous regression,
with no segmentation softmax. GroupNorm supports a one-month batch.

It uses the same v2 loss, native-coordinate Argo supervision, validation criterion,
200-epoch/500-patience budget, AdamW settings, CUDA AMP, six training loader
processes and three default seeds as the residual CNN. Parameter counts differ;
each configuration records the actual count.

```bash
bash /workspace/srtp-ocean/code/unet_baseline/run_gpu.sh
bash /workspace/srtp-ocean/code/unet_baseline/run_seeds.sh 20261117 20261118 20261119
```

The scripts start fresh runs and test the best checkpoint on 2024 and 2025-01/06.
Use `--width 24` after the single-run script to change model width. The same
environment variables, seed caveats and shared dependencies as residual CNN
apply; see `../residual_cnn/README.md` and `../climatology_baseline/README.md`.
Output prefix: `runs/unet_baseline_v2_`.

No CPU/GPU training, model forward pass or dataset evaluation was run during
delivery. CUDA runtime behavior and memory use remain untested.

2026-11-20 execution update: requested GPU runs and both test periods completed. Actual seed counts are CNN=2, U-Net=1, ViT=1; the deterministic references ran once each. Remaining repeats were cancelled at the user's request. See runs/baseline_suite_v2_20261120T105835Z/Results_README.md. No further training was started.

2026-11-20: Future runs default to at most 200 epochs. Existing early stopping and completed 250-epoch results are retained. No training launched.
