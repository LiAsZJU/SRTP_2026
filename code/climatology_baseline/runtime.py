"""Share the frozen v2 reader, CUDA-only execution and native-grid objectives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from dataset_v2.grid_sampler import GridSampler
from parallel_training.train_parallel import (
    TRAIN_KEYS,
    VALIDATION_KEYS,
    MonthCache,
    make_loader,
    training_loss,
    validate,
)
from server_training_data import TrainingDataset
from torch import nn
from torch.amp.grad_scaler import GradScaler
from train_swin_physics import Objective, append_json, atomic_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = 20261117
JsonObject = dict[str, Any]


def make_parser(architecture: str) -> argparse.ArgumentParser:
    """Build the common CUDA-only training command line.

    Args:
        architecture: One of the baseline architecture identifiers.

    Returns:
        A parser whose model-specific defaults can be overridden by its caller.
    """
    parser = argparse.ArgumentParser(description=f"Prepare or fit {architecture} on CUDA using dataset v2")
    parser.set_defaults(architecture=architecture)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data/training_ready_v2")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--validation-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--cache-gib", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-query", type=float, default=1.0)
    parser.add_argument("--lambda-physics", type=float, default=0.01)
    parser.add_argument("--stability-scale", type=float, default=1e-4)
    parser.add_argument("--stability-tolerance", type=float, default=0.01)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


def configure_cuda(seed: int, cpu_threads: int) -> torch.device:
    """Seed Python, NumPy and PyTorch, and refuse CPU fallback.

    CUDA grid_sample backward can remain nondeterministic. The run records
    best-effort determinism and allows PyTorch to warn about such operations.
    No bitwise reproducibility claim is made across GPUs or library versions.
    """
    if cpu_threads < 1 or not 0 <= seed < 2**32:
        raise ValueError("cpu-threads must be positive and seed must be in [0, 2**32)")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fitting, training and evaluation are disabled")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.device("cuda")


def load_v2_source(dataset_dir: Path) -> tuple[TrainingDataset, list[str], list[str]]:
    """Verify the agreed split and training-only normalization provenance."""
    manifest = json.loads((dataset_dir / "training_manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "swin_training_manifest_v2":
        raise ValueError("These baselines require the agreed v2 dataset manifest")
    source = TrainingDataset(dataset_dir)
    training = sorted(month for month, row in source.records.items() if row["split"] == "training")
    validation = sorted(month for month, row in source.records.items() if row["split"] == "validation")
    if len(training) != 80 or training[0] != "201504" or training[-1] != "202111":
        raise ValueError("Expected the agreed 80 training months from 201504 to 202111")
    if len(validation) != 24 or validation[0] != "202201" or validation[-1] != "202312":
        raise ValueError("Expected the agreed 24 validation months from 202201 to 202312")
    normalization = json.loads(Path(manifest["normalization"]).read_text(encoding="utf-8"))
    if sorted(normalization["fitted_months"]) != training:
        raise ValueError("Normalization was not fitted exclusively on the selected training months")
    return source, training, validation


def make_objective(source: TrainingDataset, device: torch.device, config: JsonObject) -> Objective:
    """Reuse the Swin normalization and identical neural loss coefficients."""
    return Objective(
        source,
        device,
        config["lambda_query"],
        config["lambda_physics"],
        config["stability_scale"],
        config["stability_tolerance"],
    )


def prepare_run(args: argparse.Namespace) -> tuple[TrainingDataset, torch.device, JsonObject]:
    """Check CUDA and v2 provenance before creating a fresh output directory."""
    if min(args.workers, args.validation_workers) < 0:
        raise ValueError("Worker counts must be nonnegative")
    if not math.isfinite(args.cache_gib) or args.cache_gib < 0:
        raise ValueError("cache-gib must be finite and nonnegative")
    device = configure_cuda(args.seed, args.cpu_threads)
    source, training, validation = load_v2_source(args.dataset_dir)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        {
            "schema": "ocean_baseline_checkpoint_v1",
            "dataset_manifest_sha256": hashlib.sha256(
                (args.dataset_dir / "training_manifest.json").read_bytes()
            ).hexdigest(),
            "train_months": training,
            "validation_months": validation,
            "input_channels": 276,
            "output_order": ["CT", "SA"],
            "output_shape": [2, 60, 81, 321],
            "argo_sampler": "native_coordinates",
            "determinism": "seeded_best_effort_cuda_grid_sample_backward_may_be_nondeterministic",
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        }
    )
    args.run_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.run_dir / "config.json", config)
    return source, device, config


def write_json(path: Path, payload: JsonObject) -> None:
    """Write a UTF-8 JSON artifact with finite JSON numbers only."""
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_validation(metrics: JsonObject) -> None:
    """Reject empty or nonfinite validation instead of declaring a false best."""
    if min(metrics["valid_CT_points"], metrics["valid_SA_points"]) < 1 or not math.isfinite(metrics["score"]):
        raise ValueError("Validation must contain finite errors and valid CT/SA levels")


def finish_statistical_fit(
    model: nn.Module,
    source: TrainingDataset,
    device: torch.device,
    config: JsonObject,
    fit_details: JsonObject,
) -> None:
    """Save one noniterative fit and its validation score, without gradient training."""
    pending: list[str] = []
    loader = make_loader(source, VALIDATION_KEYS, pending, config["validation_workers"], True)
    cache = MonthCache(device, config["cache_gib"])
    objective = make_objective(source, device, config)
    reference = next(iter(source.records.values()))
    sampler = GridSampler(Path(reference["background"]), device)
    metrics = validate(
        model, cache.batches(loader, config["validation_months"], pending), objective, device, False, sampler
    )
    check_validation(metrics)
    config.update(fit_details)
    run_dir = Path(config["run_dir"])
    write_json(run_dir / "config.json", config)
    epoch = int(fit_details["fit_passes"])
    record = {"epoch": epoch, "best": True, **{f"validation_{key}": value for key, value in metrics.items()}}
    append_json(run_dir / "metrics.jsonl", record)
    checkpoint = {"epoch": epoch, "model": model.state_dict(), "config": config, "best_score": metrics["score"]}
    atomic_checkpoint(run_dir / "best.pt", checkpoint)
    atomic_checkpoint(run_dir / "latest.pt", checkpoint)
    print(json.dumps(record, allow_nan=False), flush=True)


def train_neural(args: argparse.Namespace, factory: Callable[[], nn.Module]) -> None:
    """Train a neural baseline with the existing grid, Q and physics objective.

    Validation E profiles choose the checkpoint. Neither test split is read
    by this function. The launcher evaluates those splits after training.
    """
    if min(args.epochs, args.patience) < 1:
        raise ValueError("epochs and patience must be positive")
    for name in ("lr", "grad_clip", "stability_scale"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "lambda_query", "lambda_physics", "stability_tolerance"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    source, device, config = prepare_run(args)
    model = factory().to(device)
    config["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    config["fit_method"] = "AdamW_grid_plus_Argo_Q_plus_vertical_stability"
    write_json(args.run_dir / "config.json", config)
    reference = next(iter(source.records.values()))
    sampler = GridSampler(Path(reference["background"]), device)
    objective = make_objective(source, device, config)
    pending_training: list[str] = []
    pending_validation: list[str] = []
    train_loader = make_loader(source, TRAIN_KEYS, pending_training, args.workers, True)
    validation_loader = make_loader(source, VALIDATION_KEYS, pending_validation, args.validation_workers, True)
    cache = MonthCache(device, args.cache_gib)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=args.amp)
    best_score, stale = math.inf, 0
    for epoch in range(1, args.epochs + 1):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        model.train()
        order = config["train_months"].copy()
        random.Random(args.seed + epoch).shuffle(order)
        totals = torch.zeros(4, dtype=torch.float64, device=device)
        skipped_steps = 0
        for step, (month, sample) in enumerate(cache.batches(train_loader, order, pending_training), 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                prediction = cast(torch.Tensor, model(sample["input"]))
            losses = training_loss(prediction.float(), sample, objective, sampler)
            loss = losses[0]
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite loss at epoch={epoch}, month={month}")
            previous_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            skipped_steps += int(scaler.get_scale() < previous_scale)
            totals += torch.stack([value.detach().double() for value in losses])
            if step % 10 == 0 or step == len(order):
                print(f"epoch={epoch} step={step}/{len(order)} month={month} loss={loss.item():.6f}", flush=True)
        metrics = validate(
            model,
            cache.batches(validation_loader, config["validation_months"], pending_validation),
            objective,
            device,
            args.amp,
            sampler,
        )
        check_validation(metrics)
        score = float(metrics["score"])
        improved = score < best_score
        best_score, stale = (score, 0) if improved else (best_score, stale + 1)
        torch.cuda.synchronize(device)
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            **dict(
                zip(
                    ("train_total", "train_grid", "train_query", "train_physics"),
                    (totals / len(order)).tolist(),
                    strict=True,
                )
            ),
            **{f"validation_{key}": value for key, value in metrics.items()},
            "amp_skipped_steps": skipped_steps,
            "best": improved,
        }
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "best_score": best_score,
            "stale": stale,
        }
        append_json(args.run_dir / "metrics.jsonl", record)
        atomic_checkpoint(args.run_dir / "latest.pt", checkpoint)
        if improved:
            atomic_checkpoint(args.run_dir / "best.pt", checkpoint)
        print(json.dumps(record, allow_nan=False), flush=True)
        if stale >= args.patience:
            break
