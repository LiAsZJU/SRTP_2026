# Vertical EOF plus ridge regression

A statistical reference with a shared local mapping from the same 276 input
channels to joint 120-dimensional CT/SA columns. This is a vertical EOF model,
not a global spatial EOF flattened across the complete ocean domain.

1. Read only the 80 training months.
2. Select grid columns where both CT and SA are valid at all 60 depths. Missing
   targets are never counted as zero. Shallow or incomplete columns are excluded
   from fitting; their exclusion counts are saved per month.
3. Accumulate area-weighted float64 covariances on CUDA in chunks of 8192 rows.
4. Obtain 16 leading joint vertical EOF modes using `torch.linalg.eigh`.
5. Fit standardized local input-to-mode ridge coefficients with alpha 0.01 using
   `torch.linalg.solve`. Reconstruct all columns from their predicted modes.

No iterative epochs, random initialization, Argo Q loss or physical penalty are
used. The checkpoint records one fitting pass (epoch 1), not 500 training epochs.
Input C profiles and masks remain available through the existing 276 channels.
It is a statistical reference with a different fitting objective; CNN/U-Net/Swin
are the controlled architecture comparison with the same supervision losses.

```bash
bash /workspace/srtp-ocean/code/eof_ridge/run_gpu.sh --modes 16 --ridge-alpha 0.01
bash /workspace/srtp-ocean/code/eof_ridge/run_seeds.sh 20261117 20261118 20261119
```

Choose modes and alpha on validation only. The default launcher tests both test
splits afterward; do not inspect test scores repeatedly to tune these settings.
This closed-form model normally needs one fit. Repeated seeds are deterministic
controls; variation cannot be interpreted as initialization sensitivity.

See `../climatology_baseline/README.md` for shared dependencies, the CUDA guard,
artifact structure and evaluation protocol. Output prefix: `runs/eof_ridge_v2_`.
CUDA fitting and data evaluation have not been run during this delivery.

2026-11-20 execution update: requested GPU runs and both test periods completed. Actual seed counts are CNN=2, U-Net=1, ViT=1; the deterministic references ran once each. Remaining repeats were cancelled at the user's request. See runs/baseline_suite_v2_20261120T105835Z/Results_README.md. No further training was started.
