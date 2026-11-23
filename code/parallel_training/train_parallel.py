#!/usr/bin/env python3
"""Train the original Swin model with parallel monthly data loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from dataset_v2.grid_sampler import GridSampler
from server_training_data import ROOT as BASELINE_DATASET
from server_training_data import TrainingDataset
from train_swin_physics import (
    Objective,
    SwinReconstructor,
    append_json,
    atomic_checkpoint,
    masked_mean,
    model_input,
)

RUNS_DIRECTORY = Path(__file__).resolve().parents[2] / "runs"
MonthTensors = dict[str, torch.Tensor]
MonthBatch = tuple[str, MonthTensors]
PROFILE_KEYS = ("profile_position", "profile_role_code", "profile_mask", "profile_values")
TRAIN_KEYS = PROFILE_KEYS + ("target", "target_mask", "area_weight", "background", "alpha", "beta")
VALIDATION_KEYS = PROFILE_KEYS + ("profile_background", "profile_absolute")


def seed_everything(seed: int, use_cuda: bool) -> None:
    """Seed every training RNG before constructing the model and loaders.

    Args:
        seed: Integer in the NumPy-compatible range [0, 2**32).
        use_cuda: Whether to seed CUDA devices as well as the CPU generator.

    Raises:
        ValueError: If the seed is outside the supported range.

    Notes:
        CUDA grid_sample backward can be nondeterministic. Use best-effort
        deterministic algorithms with warnings, rather than claiming exact
        reproducibility or failing the existing Argo supervision operation.
    """
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Seed a loader worker's Python and NumPy RNGs from PyTorch's worker seed.

    Args:
        worker_id: Loader worker index, already included in torch.initial_seed().
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class MonthlySamples(Dataset[MonthBatch]):
    """Prepare one month at a time using the existing NetCDF reader."""

    def __init__(self, source: TrainingDataset, keys: tuple[str, ...]) -> None:
        """Set the shared data source and the fields needed for one phase."""
        self.source = source
        self.keys = keys

    def __getitem__(self, month: str) -> MonthBatch:
        """Load a month and pack its model input and supervision tensors."""
        sample = self.source.load(month)
        tensors = {key: torch.from_numpy(cast(NDArray[np.generic], sample[key])) for key in self.keys}
        tensors["input"] = model_input(sample, torch.device("cpu"))
        return month, tensors

    def __len__(self) -> int:
        """Return the number of months in the source manifest."""
        return len(self.source.records)


class MonthCache:
    """Keep a bounded set of prepared months on the training device."""

    def __init__(self, device: torch.device, capacity_gib: float) -> None:
        """Set the destination device and the cache limit in GiB."""
        self.device = device
        self.capacity = int(capacity_gib * 1024**3)
        self.used = 0
        self.samples: dict[str, MonthTensors] = {}

    def batches(
        self,
        loader: Iterable[MonthBatch],
        order: list[str],
        pending: list[str],
    ) -> Iterator[MonthBatch]:
        """Yield device tensors in the requested month order.

        Args:
            loader: CPU loader whose sampler reads the pending list.
            order: Month identifiers in the order expected by training or validation.
            pending: Mutable sampler input updated with the months missing from the cache.

        Yields:
            The month identifier and its tensors on the destination device.

        Raises:
            RuntimeError: If the loader yields a different month than expected.
        """
        pending[:] = [month for month in order if month not in self.samples]
        uncached_batches = iter(loader) if pending else iter(())
        for month in order:
            if month in self.samples:
                yield month, self.samples[month]
                continue
            loaded_month, cpu_sample = next(uncached_batches)
            if loaded_month != month:
                raise RuntimeError(f"Expected {month}, loaded {loaded_month}")
            device_sample = {
                key: value.to(self.device, non_blocking=self.device.type == "cuda") for key, value in cpu_sample.items()
            }
            size = sum(value.numel() * value.element_size() for value in device_sample.values())
            if self.used + size <= self.capacity:
                if self.device.type == "cpu":
                    # Release DataLoader shared-memory descriptors after the batch.
                    device_sample = {
                        # PyTorch 2.7 does not annotate Tensor.is_shared.
                        key: value.clone() if value.is_shared() else value  # type: ignore[no-untyped-call]
                        for key, value in device_sample.items()
                    }
                self.samples[month] = device_sample
                self.used += size
            yield month, device_sample


def training_loss(
    prediction: torch.Tensor, sample: MonthTensors, objective: Objective, sampler: GridSampler
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the original grid, Argo Q, and vertical stability objectives.

    Args:
        prediction: Normalized CT and SA fields for one month.
        sample: Training targets, masks, and auxiliary tensors on the same device.
        objective: Fitted normalization and configured loss weights.
        sampler: Observation operator for the native CORA axes.

    Returns:
        The total, grid, query, and physics losses, in that order.
    """
    target, target_mask = sample["target"][None], sample["target_mask"][None]
    area = sample["area_weight"][None, None, None]
    grid = masked_mean((prediction - target).square(), target_mask, area)

    profiles = sampler(prediction, sample["profile_position"])
    query_mask = sample["profile_mask"] * (sample["profile_role_code"] == 1)[:, None, None]
    query_error = F.smooth_l1_loss(profiles, sample["profile_values"], reduction="none", beta=0.5)
    query = masked_mean(query_error, query_mask)

    physical = sample["background"][None] + prediction * objective.std + objective.mean
    ct, sa = physical[:, 0], physical[:, 1]
    alpha, beta = sample["alpha"][None], sample["beta"][None]
    alpha_mid = 0.5 * (alpha[:, 1:] + alpha[:, :-1])
    beta_mid = 0.5 * (beta[:, 1:] + beta[:, :-1])
    proxy = (beta_mid * (sa[:, 1:] - sa[:, :-1]) - alpha_mid * (ct[:, 1:] - ct[:, :-1])) / objective.stability_scale
    valid = target_mask[:, :, 1:] * target_mask[:, :, :-1]
    valid = valid[:, 0] * valid[:, 1] * (alpha_mid != 0) * (beta_mid != 0)
    physics = masked_mean(F.relu(-proxy - objective.stability_tolerance).square(), valid, area[:, 0, 0])
    total = grid + objective.lambda_query * query + objective.lambda_physics * physics
    return total, grid, query, physics


@torch.no_grad()
def validate(
    model: nn.Module,
    batches: Iterable[MonthBatch],
    objective: Objective,
    device: torch.device,
    use_amp: bool,
    sampler: GridSampler,
) -> dict[str, float | int]:
    """Evaluate the model against held-out Argo E profiles.

    Args:
        model: Swin reconstructor in the current training run.
        batches: Validation months with their tensors on the destination device.
        objective: Fitted normalization used to recover physical CT and SA.
        device: Device used for predictions and metric accumulation.
        use_amp: Whether to use automatic mixed precision for the forward pass.
        sampler: Observation operator for the native CORA axes.

    Returns:
        Normalized score, physical RMSE by variable, and valid point counts.
    """
    model.eval()
    totals = torch.zeros(6, device=device, dtype=torch.float64)
    mean, std = objective.mean[0, :, :, 0, 0], objective.std[0, :, :, 0, 0]
    for _, sample in batches:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(sample["input"])
        profiles = sampler(prediction.float(), sample["profile_position"])
        mask = sample["profile_mask"].bool() & (sample["profile_role_code"] == 2)[:, None, None]
        normalized_sum = torch.where(mask, (profiles - sample["profile_values"]).square(), 0).sum()
        physical = sample["profile_background"] + profiles * std + mean
        physical_sums = []
        physical_counts = []
        for variable in range(2):
            variable_mask = mask[:, variable]
            error = physical[:, variable] - sample["profile_absolute"][:, variable]
            physical_sums.append(torch.where(variable_mask, error.square(), 0).sum().double())
            physical_counts.append(variable_mask.sum().double())
        totals += torch.stack(
            (
                normalized_sum.double(),
                mask.sum().double(),
                *physical_sums,
                *physical_counts,
            )
        )
    normalized_sum, count, ct_sum, sa_sum, ct_count, sa_count = totals.tolist()
    return {
        "score": math.sqrt(normalized_sum / max(count, 1)),
        "CT_RMSE_C": math.sqrt(ct_sum / max(ct_count, 1)),
        "SA_RMSE_g_kg": math.sqrt(sa_sum / max(sa_count, 1)),
        "valid_CT_points": int(ct_count),
        "valid_SA_points": int(sa_count),
    }


def parse_args() -> argparse.Namespace:
    """Read execution options and original model defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUNS_DIRECTORY / "swin_physics_parallel_v1")
    parser.add_argument("--dataset-dir", type=Path, default=BASELINE_DATASET)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--cache-gib", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-query", type=float, default=1.0)
    parser.add_argument("--lambda-physics", type=float, default=0.01)
    parser.add_argument("--stability-scale", type=float, default=1e-4)
    parser.add_argument("--stability-tolerance", type=float, default=0.01)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-train-months", type=int)
    parser.add_argument("--max-validation-months", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(args.workers, args.validation_workers) < 0 or args.cpu_threads < 1:
        parser.error("worker counts must be nonnegative and cpu-threads must be positive")
    if not math.isfinite(args.cache_gib) or args.cache_gib < 0:
        parser.error("cache-gib must be finite and nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    positive = (args.epochs, args.patience, args.log_every, args.max_train_months, args.max_validation_months)
    if any(value is not None and value < 1 for value in positive):
        parser.error("epochs, patience, log-every and month limits must be positive")
    return args


def make_loader(
    source: TrainingDataset,
    keys: tuple[str, ...],
    pending: list[str],
    workers: int,
    pin_memory: bool,
    seed: int | None = None,
) -> DataLoader[MonthBatch]:
    """Create a loader that preserves the order of requested months.

    Args:
        source: NetCDF reader shared by the dataset wrapper.
        keys: Sample fields to return together with the model input.
        pending: Mutable sampler input populated before each pass over the loader.
        workers: Number of spawned loader processes. Zero loads in the main process.
        pin_memory: Whether to stage tensors in page-locked CPU memory.
        seed: Optional worker-base RNG seed. None preserves existing callers.

    Returns:
        A loader yielding one requested month at a time.
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        MonthlySamples(source, keys),
        sampler=pending,
        batch_size=None,
        num_workers=workers,
        pin_memory=pin_memory,
        multiprocessing_context="spawn" if workers else None,
        persistent_workers=bool(workers),
        prefetch_factor=1 if workers else None,
        generator=generator,
        worker_init_fn=seed_worker if seed is not None else None,
    )


def train(args: argparse.Namespace) -> None:
    """Train the reconstructor and write metrics and checkpoints for a fresh run.

    Args:
        args: Options parsed by :func:`parse_args`.

    Raises:
        RuntimeError: If CUDA was requested and is unavailable.
        ValueError: If the selected data has no training or validation month.
        FileExistsError: If the output directory already exists.
    """
    device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is visible")
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed, use_cuda)
    source = TrainingDataset(args.dataset_dir)
    reference_record = next(iter(source.records.values()))
    sampler = GridSampler(Path(reference_record["background"]), device)
    manifest_digest = hashlib.sha256((args.dataset_dir / "training_manifest.json").read_bytes()).hexdigest()
    train_months = [month for month, row in source.records.items() if row["split"] == "training"]
    validation_months = [month for month, row in source.records.items() if row["split"] == "validation"]
    train_months = train_months[: args.max_train_months]
    validation_months = validation_months[: args.max_validation_months]
    if not train_months or not validation_months:
        raise ValueError("At least one training and one validation month are required")
    pending_train: list[str] = []
    pending_validation: list[str] = []
    train_loader = make_loader(source, TRAIN_KEYS, pending_train, args.workers, use_cuda, seed=args.seed)
    validation_loader = make_loader(
        source,
        VALIDATION_KEYS,
        pending_validation,
        args.validation_workers,
        use_cuda,
        seed=args.seed + 1,
    )
    cache = MonthCache(device, args.cache_gib)
    model = SwinReconstructor(276, args.dim, args.blocks, args.heads, args.window).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp and use_cuda
    scaler = GradScaler("cuda", enabled=use_amp)
    objective = Objective(
        source,
        device,
        args.lambda_query,
        args.lambda_physics,
        args.stability_scale,
        args.stability_tolerance,
    )
    args.run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "run_dir": str(args.run_dir),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_manifest_sha256": manifest_digest,
        "argo_sampler": "native_coordinates",
        "random_seed_scope": ["python", "numpy", "torch_cpu", "torch_cuda", "data_loader_workers"],
        "worker_seed_rule": "torch_initial_seed_modulo_2**32",
        "determinism": "best_effort_warn_only_cuda_grid_sample_backward_may_be_nondeterministic",
        "device_resolved": str(device),
        "input_channels": 276,
        "train_months": train_months,
        "validation_months": validation_months,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "device": str(device),
                "workers": args.workers,
                "validation_workers": args.validation_workers,
                "train_months": len(train_months),
                "validation_months": len(validation_months),
            }
        ),
        flush=True,
    )
    best_score, stale = math.inf, 0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        order = train_months.copy()
        random.Random(args.seed + epoch).shuffle(order)
        totals = torch.zeros(4, device=device, dtype=torch.float64)
        batches = cache.batches(train_loader, order, pending_train)
        for step, (month, sample) in enumerate(batches, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(sample["input"])
                losses = training_loss(prediction, sample, objective, sampler)
            loss = losses[0]
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]  # PyTorch 2.7 stub limitation.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            totals += torch.stack([value.detach().double() for value in losses])
            if step % args.log_every == 0 or step == len(order):
                print(f"epoch={epoch} step={step}/{len(order)} month={month} loss={loss.item():.6f}", flush=True)

        validation_batches = cache.batches(validation_loader, validation_months, pending_validation)
        metrics = validate(model, validation_batches, objective, device, use_amp, sampler)
        improved = metrics["score"] < best_score
        best_score, stale = (metrics["score"], 0) if improved else (best_score, stale + 1)
        train_metrics = dict(zip(("total", "grid", "query", "physics"), (totals / len(order)).tolist(), strict=True))
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in metrics.items()},
            "netcdf_train_months": len(pending_train),
            "netcdf_validation_months": len(pending_validation),
            "cached_months": len(cache.samples),
            "cache_gib": round(cache.used / 1024**3, 3),
            "best": improved,
        }
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best_score,
            "stale": stale,
            "config": config,
        }
        append_json(args.run_dir / "metrics.jsonl", record)
        atomic_checkpoint(args.run_dir / "latest.pt", checkpoint)
        if improved:
            atomic_checkpoint(args.run_dir / "best.pt", checkpoint)
        print(json.dumps(record), flush=True)
        if stale >= args.patience:
            print(f"early_stop epoch={epoch} patience={args.patience}", flush=True)
            break


if __name__ == "__main__":
    train(parse_args())
