"""Fit a joint vertical EOF and ridge regression with CUDA sufficient statistics."""

from __future__ import annotations

import math
import time

import torch
from climatology_baseline.runtime import finish_statistical_fit, make_parser, prepare_run
from parallel_training.train_parallel import MonthCache, make_loader

from eof_ridge.model import EOFRidge


@torch.no_grad()
def main() -> None:
    """Fit training CORA columns only, then score the fixed model on validation E.

    One training row is one grid column in one training month. All 120 CT/SA
    targets must be valid. Latitude-area weights are applied to the moments.
    Masks are input features; missing targets are never treated as zero data.
    The basis is vertical, not a flattened full-domain spatial EOF.
    """
    parser = make_parser("eof_ridge")
    parser.set_defaults(epochs=0, patience=0, amp=False)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--ridge-alpha", type=float, default=0.01)
    parser.add_argument("--chunk-rows", type=int, default=8192)
    args = parser.parse_args()
    if not 1 <= args.modes <= 120 or args.chunk_rows < 1:
        parser.error("modes must be in [1, 120] and chunk-rows must be positive")
    if not math.isfinite(args.ridge_alpha) or args.ridge_alpha <= 0:
        parser.error("ridge-alpha must be finite and positive")
    source, device, config = prepare_run(args)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    # Accumulate raw moments in float64; no whole-period tensor is held in memory.
    total_weight = torch.zeros((), dtype=torch.float64, device=device)
    input_sum = torch.zeros(276, dtype=torch.float64, device=device)
    target_sum = torch.zeros(120, dtype=torch.float64, device=device)
    input_product = torch.zeros(276, 276, dtype=torch.float64, device=device)
    cross_product = torch.zeros(276, 120, dtype=torch.float64, device=device)
    target_product = torch.zeros(120, 120, dtype=torch.float64, device=device)
    pending: list[str] = []
    loader = make_loader(source, ("target", "target_mask", "area_weight"), pending, args.workers, True)
    cache = MonthCache(device, 0)
    fitted_columns = 0
    eligible_by_month: dict[str, int] = {}
    for month, sample in cache.batches(loader, config["train_months"], pending):
        valid = sample["target_mask"].bool().all(dim=(0, 1)).reshape(-1)
        inputs = sample["input"][0].permute(1, 2, 0).reshape(-1, 276)[valid]
        targets = sample["target"].reshape(120, -1).T[valid]
        weights = sample["area_weight"].reshape(-1)[valid]
        count = inputs.shape[0]
        fitted_columns += count
        eligible_by_month[month] = count
        for start in range(0, count, args.chunk_rows):
            x = inputs[start : start + args.chunk_rows].double()
            y = targets[start : start + args.chunk_rows].double()
            weight = weights[start : start + args.chunk_rows].double()[:, None]
            total_weight += weight.sum()
            input_sum += (x * weight).sum(dim=0)
            target_sum += (y * weight).sum(dim=0)
            input_product += x.T @ (x * weight)
            cross_product += x.T @ (y * weight)
            target_product += y.T @ (y * weight)
        print(f"fit month={month} complete_columns={count}", flush=True)
    if fitted_columns <= 276 or total_weight.item() <= 0:
        raise ValueError("Insufficient complete training columns for the statistical baseline")
    mean_x = input_sum / total_weight
    mean_y = target_sum / total_weight
    covariance_x = input_product / total_weight - torch.outer(mean_x, mean_x)
    covariance_y = target_product / total_weight - torch.outer(mean_y, mean_y)
    covariance_xy = cross_product / total_weight - torch.outer(mean_x, mean_y)
    scale = covariance_x.diag().clamp_min(1e-12).sqrt()
    eigenvalues, eigenvectors = torch.linalg.eigh((covariance_y + covariance_y.T) / 2)
    basis = eigenvectors[:, -args.modes :].flip(dims=(1,))
    standardized_covariance = covariance_x / torch.outer(scale, scale)
    system = (standardized_covariance + standardized_covariance.T) / 2
    system += args.ridge_alpha * torch.eye(276, dtype=torch.float64, device=device)
    coefficients = torch.linalg.solve(system, (covariance_xy / scale[:, None]) @ basis)
    model = EOFRidge(args.modes).to(device)
    model.input_mean.copy_(mean_x.float())
    model.input_scale.copy_(scale.float())
    model.target_mean.copy_(mean_y.float())
    model.basis.copy_(basis.float())
    model.coefficients.copy_(coefficients.float())
    torch.cuda.synchronize(device)
    energy = eigenvalues.clamp_min(0)
    finish_statistical_fit(
        model,
        source,
        device,
        config,
        {
            "fit_passes": 1,
            "fit_seconds": time.perf_counter() - started,
            "fit_method": "area_weighted_joint_vertical_EOF_plus_standardized_local_ridge",
            "fitted_columns": fitted_columns,
            "complete_columns_by_month": eligible_by_month,
            "retained_training_variance_fraction": float((energy[-args.modes :].sum() / energy.sum()).item()),
            "target_missing_rule": "fit_only_columns_valid_for_both_variables_at_all_60_depths",
            "uses_Argo_Q_loss": False,
            "uses_physics_loss": False,
            "stochastic_model": False,
        },
    )


if __name__ == "__main__":
    main()
