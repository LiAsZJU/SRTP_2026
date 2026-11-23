#!/usr/bin/env python3
"""Train and evaluate v4 Swin / Residual CNN models with Argo C Context Dropout.

This entry point preserves existing baseline files (`train_parallel.py`, `train_arch.py`)
unchanged while adding:
  1. Per-step, non-mutating Argo C Context Dropout after GPU `MonthCache` lookup.
  2. Strict v4 `no_smap` enforcement (verifies SMAP channels `12:15` and `27:30` are zero).
  3. Complete `config.json` provenance (`arch`, `ablate_mode`, `context_dropout`, manifest SHA-256).
  4. Automatic post-training evaluation on `test_locked` (2022–2024) and `forward_test` (2025H1),
     reporting both `all_E` and `E_platforms_unseen_in_validation` (plus optional `no_context` score).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
PARALLEL_DIR = Path(__file__).resolve().parent
if str(PARALLEL_DIR) not in sys.path:
    sys.path.insert(0, str(PARALLEL_DIR))

from context_dropout import (  # noqa: E402
    VALID_CONTEXT_DROPOUT_MODES,
    ContextDropoutConfig,
    apply_context_dropout,
    assert_no_smap_input,
    zero_all_context,
)
from dataset_v2.evaluate_checkpoint import (  # noqa: E402
    Metrics,
    eligible_validation_platforms,
    load_profiles,
)
from dataset_v2.grid_sampler import GridSampler  # noqa: E402
from residual_cnn.model import ResidualCNN  # noqa: E402
from server_training_data import TrainingDataset  # noqa: E402
from train_parallel import (  # noqa: E402
    TRAIN_KEYS,
    VALIDATION_KEYS,
    MonthCache,
    make_loader,
    seed_everything,
    training_loss,
    validate,
)
from train_swin_physics import Objective, SwinReconstructor, append_json, atomic_checkpoint, model_input  # noqa: E402
from unet_baseline.model import UNetReconstructor  # noqa: E402
from vit_baseline.model import ViTReconstructor  # noqa: E402

DEFAULT_V4_DATASET = Path("/g3240101584lt/srtp-ocean/data/training_ready_v4")
RUNS_DIRECTORY = Path(__file__).resolve().parents[2] / "runs"


def build_model(
    arch: str,
    in_channels: int,
    dim: int,
    blocks: int,
    heads: int,
    window: int,
) -> nn.Module:
    """Instantiate the requested architecture with exact baseline parameter mapping."""
    if arch == "swin":
        return SwinReconstructor(in_channels, dim, blocks, heads, window)
    if arch == "cnn":
        return ResidualCNN(width=dim, blocks=blocks)
    if arch == "unet":
        return UNetReconstructor(width=dim)
    if arch == "vit":
        return ViTReconstructor(embedding_dim=dim, blocks=blocks, heads=heads, dropout=0.0)
    raise ValueError(f"Unsupported architecture: {arch}")


def parse_args() -> argparse.Namespace:
    """Parse CLI options for v4 Context Dropout training and evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=("swin", "cnn", "unet", "vit"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_V4_DATASET)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--validation-workers", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--cache-gib", type=float, default=14.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=201)
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
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-train-months", type=int)
    parser.add_argument("--max-validation-months", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    # Context dropout options
    parser.add_argument("--context-drop-mode", choices=VALID_CONTEXT_DROPOUT_MODES, default="mixed")
    parser.add_argument("--context-drop-p-clean", type=float, default=0.15)
    parser.add_argument("--context-drop-p-full", type=float, default=0.20)
    parser.add_argument("--context-drop-p-block", type=float, default=0.35)
    parser.add_argument("--context-drop-min-keep", type=float, default=0.10)
    parser.add_argument("--context-drop-max-keep", type=float, default=0.85)
    parser.add_argument("--context-drop-num-blocks-min", type=int, default=1)
    parser.add_argument("--context-drop-num-blocks-max", type=int, default=3)
    parser.add_argument("--context-drop-block-h-min", type=int, default=10)
    parser.add_argument("--context-drop-block-h-max", type=int, default=28)
    parser.add_argument("--context-drop-block-w-min", type=int, default=16)
    parser.add_argument("--context-drop-block-w-max", type=int, default=48)

    # Safety & post-training evaluation options
    parser.add_argument("--require-no-smap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-no-context", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    if min(args.workers, args.validation_workers) < 0 or args.cpu_threads < 1:
        parser.error("worker counts must be nonnegative and cpu-threads must be positive")
    if not math.isfinite(args.cache_gib) or args.cache_gib < 0:
        parser.error("cache-gib must be finite and nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    return args


def make_dropout_config(args: argparse.Namespace) -> ContextDropoutConfig:
    """Build and validate ContextDropoutConfig from parsed CLI arguments."""
    return ContextDropoutConfig(
        mode=args.context_drop_mode,
        p_clean=args.context_drop_p_clean,
        p_full=args.context_drop_p_full,
        p_block=args.context_drop_p_block,
        min_keep=args.context_drop_min_keep,
        max_keep=args.context_drop_max_keep,
        num_blocks_min=args.context_drop_num_blocks_min,
        num_blocks_max=args.context_drop_num_blocks_max,
        block_h_min=args.context_drop_block_h_min,
        block_h_max=args.context_drop_block_h_max,
        block_w_min=args.context_drop_block_w_min,
        block_w_max=args.context_drop_block_w_max,
    )


def verify_v4_environment(require_no_smap: bool) -> str:
    """Validate ABLATE_MODE consistency before training or evaluation."""
    ablate_mode = os.environ.get("ABLATE_MODE", "full").strip()
    if require_no_smap and ablate_mode != "no_smap":
        raise RuntimeError(
            f"v4 training requires ABLATE_MODE=no_smap (got {ablate_mode!r}). "
            "Export ABLATE_MODE=no_smap and put parallel_training/ablate_site first on PYTHONPATH."
        )
    return ablate_mode


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_dir: Path,
    output_path: Path,
    device: torch.device,
    splits: tuple[str, ...] = ("test_locked", "forward_test"),
    zero_context: bool = False,
    require_no_smap: bool = True,
) -> dict[str, object]:
    """Evaluate a saved checkpoint on test splits with full and unseen E platform cohorts.

    Args:
        checkpoint_path: Path to ``best.pt``.
        dataset_dir: Dataset directory containing ``training_manifest.json``.
        output_path: Destination JSON file (must not already exist).
        device: Target torch device.
        splits: Dataset splits to evaluate.
        zero_context: If True, zero all Argo C channels (`30:276`) during inference.
        require_no_smap: If True, assert SMAP input channels are strictly zeroed.

    Returns:
        Structured evaluation report dictionary.
    """
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output_path}")

    verify_v4_environment(require_no_smap)
    source = TrainingDataset(dataset_dir)
    manifest_hash = hashlib.sha256((dataset_dir / "training_manifest.json").read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    expected_hash = config.get("dataset_manifest_sha256")
    if expected_hash is not None and expected_hash != manifest_hash:
        raise ValueError("Checkpoint and dataset manifest SHA-256 do not match")

    arch = str(config["arch"])
    model = build_model(
        arch=arch,
        in_channels=int(config["input_channels"]),
        dim=int(config["dim"]),
        blocks=int(config["blocks"]),
        heads=int(config["heads"]),
        window=int(config["window"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    sampler = GridSampler(Path(next(iter(source.records.values()))["background"]), device)
    objective = Objective(
        source,
        device,
        float(config["lambda_query"]),
        float(config["lambda_physics"]),
        float(config["stability_scale"]),
        float(config["stability_tolerance"]),
    )
    mean = objective.mean[0, :, :, 0, 0]
    std = objective.std[0, :, :, 0, 0]
    validation_platforms = sorted(eligible_validation_platforms(source))

    report: dict[str, object] = {
        "checkpoint": str(checkpoint_path.resolve()),
        "architecture": arch,
        "ablate_mode": os.environ.get("ABLATE_MODE", "full").strip(),
        "eval_zero_context": zero_context,
        "seed": config["seed"],
        "epoch": checkpoint["epoch"],
        "dataset": str(dataset_dir.resolve()),
        "manifest_sha256": manifest_hash,
        "device": str(device),
        "metrics": {},
    }

    metrics_by_split: dict[str, object] = {}
    for split in splits:
        months = sorted(month for month, row in source.records.items() if row["split"] == split)
        if not months:
            raise ValueError(f"Empty or unknown split: {split}")
        cohort_names = ["all_E"]
        if split != "validation":
            cohort_names.append("E_platforms_unseen_in_validation")
        accumulators = {name: Metrics() for name in cohort_names}

        for month in months:
            sample = source.load(month)
            roles, platforms = load_profiles(source.records[month]["argo_profiles"])
            cohorts = {
                "all_E": roles == 2,
                "E_platforms_unseen_in_validation": (roles == 2) & ~np.isin(platforms, validation_platforms),
            }
            inputs = model_input(sample, device)
            if require_no_smap:
                assert_no_smap_input(inputs)
            if zero_context:
                inputs = zero_all_context(inputs)

            use_amp = device.type == "cuda" and bool(config["amp"])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                predicted_grid = cast(torch.Tensor, model(inputs))

            positions = torch.from_numpy(sample["profile_position"]).to(device=device, dtype=torch.float32)
            background = torch.from_numpy(sample["profile_background"]).to(device=device, dtype=torch.float32)
            normalized = sampler(predicted_grid.float(), positions)
            physical = (background + normalized * std + mean).cpu().numpy()
            errors = normalized.cpu().numpy() - sample["profile_values"]
            valid = sample["profile_mask"].astype(bool)
            for cohort_name, accum in accumulators.items():
                accum.add(
                    sample["profile_absolute"],
                    physical,
                    errors,
                    valid,
                    cohorts[cohort_name],
                    platforms,
                )

        metrics_by_split[split] = {
            "months": months,
            "operators": {
                "native_coordinates": {name: item.summary() for name, item in accumulators.items()},
            },
        }
        print(
            f"[eval zero_context={zero_context}] split={split} all_E={accumulators['all_E'].summary()}",
            flush=True,
        )

    report["metrics"] = metrics_by_split
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def train(args: argparse.Namespace) -> None:
    """Run v4 training with dynamic Argo C Context Dropout and post-train evaluation."""
    ablate_mode = verify_v4_environment(args.require_no_smap)
    dropout_cfg = make_dropout_config(args)

    use_cuda = args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    seed_everything(args.seed, use_cuda)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)

    source = TrainingDataset(args.dataset_dir)
    manifest_digest = hashlib.sha256((args.dataset_dir / "training_manifest.json").read_bytes()).hexdigest()
    train_months = [month for month, row in source.records.items() if row["split"] == "train"]
    validation_months = [month for month, row in source.records.items() if row["split"] == "validation"]
    if args.max_train_months is not None:
        train_months = train_months[: args.max_train_months]
    if args.max_validation_months is not None:
        validation_months = validation_months[: args.max_validation_months]

    reference = next(iter(source.records.values()))
    sampler = GridSampler(Path(reference["background"]), device)
    pending_train: list[str] = []
    pending_validation: list[str] = []
    train_loader = make_loader(
        source,
        TRAIN_KEYS,
        pending_train,
        args.workers,
        use_cuda,
        seed=args.seed,
    )
    validation_loader = make_loader(
        source,
        VALIDATION_KEYS,
        pending_validation,
        args.validation_workers,
        use_cuda,
        seed=args.seed + 1,
    )
    cache = MonthCache(device, args.cache_gib)

    model = build_model(args.arch, 276, args.dim, args.blocks, args.heads, args.window).to(device)
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
    config: dict[str, object] = {
        **vars(args),
        "schema": "ocean_v4_context_dropout_v1",
        "run_dir": str(args.run_dir),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_manifest_sha256": manifest_digest,
        "ablate_mode": ablate_mode,
        "context_dropout": dropout_cfg.to_dict(),
        "argo_sampler": "native_coordinates",
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
                "arch": args.arch,
                "ablate_mode": ablate_mode,
                "context_dropout": dropout_cfg.to_dict(),
                "device": str(device),
                "train_months": len(train_months),
                "validation_months": len(validation_months),
                "parameters": config["parameters"],
            }
        ),
        flush=True,
    )

    best_score, stale = math.inf, 0
    smap_checked_train = False
    smap_checked_val = False

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        order = train_months.copy()
        random.Random(args.seed + epoch).shuffle(order)
        aug_generator = torch.Generator(device="cpu").manual_seed((args.seed + 1_000_003 + epoch) % (2**32))

        totals = torch.zeros(4, device=device, dtype=torch.float64)
        kept_frac_sum = 0.0
        regime_counts = {"clean": 0, "pointwise": 0, "block": 0, "full_drop": 0}

        batches = cache.batches(train_loader, order, pending_train)
        for step, (month, sample) in enumerate(batches, 1):
            if args.require_no_smap and not smap_checked_train:
                assert_no_smap_input(sample["input"])
                smap_checked_train = True

            # Apply dynamic dropout to a cloned tensor; keep cache.samples untouched.
            aug_input, kept_frac, regime = apply_context_dropout(sample["input"], dropout_cfg, aug_generator)
            kept_frac_sum += kept_frac
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(aug_input)
                losses = training_loss(prediction, sample, objective, sampler)
            loss = losses[0]
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            totals += torch.stack([value.detach().double() for value in losses])
            if step % args.log_every == 0 or step == len(order):
                print(
                    f"epoch={epoch} step={step}/{len(order)} month={month} "
                    f"regime={regime} kept={kept_frac:.3f} loss={loss.item():.6f}",
                    flush=True,
                )

        # Validation ALWAYS uses unmodified full context (no Context Dropout).
        val_batches_iter = cache.batches(validation_loader, validation_months, pending_validation)
        if args.require_no_smap and not smap_checked_val:
            checked_batches = []
            for month, val_sample in val_batches_iter:
                assert_no_smap_input(val_sample["input"])
                checked_batches.append((month, val_sample))
            smap_checked_val = True
            metrics = validate(model, checked_batches, objective, device, use_amp, sampler)
        else:
            metrics = validate(model, val_batches_iter, objective, device, use_amp, sampler)

        improved = metrics["score"] < best_score
        best_score, stale = (metrics["score"], 0) if improved else (best_score, stale + 1)
        train_metrics = dict(zip(("total", "grid", "query", "physics"), (totals / len(order)).tolist(), strict=True))
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "train_kept_context_frac": kept_frac_sum / max(len(order), 1),
            "train_regimes": regime_counts,
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

    best_ckpt = args.run_dir / "best.pt"
    if args.evaluate and best_ckpt.is_file():
        evaluate_checkpoint(
            checkpoint_path=best_ckpt,
            dataset_dir=args.dataset_dir,
            output_path=args.run_dir / "test_metrics.json",
            device=device,
            splits=("test_locked", "forward_test"),
            zero_context=False,
            require_no_smap=args.require_no_smap,
        )
        if args.eval_no_context:
            evaluate_checkpoint(
                checkpoint_path=best_ckpt,
                dataset_dir=args.dataset_dir,
                output_path=args.run_dir / "test_metrics_no_context.json",
                device=device,
                splits=("test_locked", "forward_test"),
                zero_context=True,
                require_no_smap=args.require_no_smap,
            )


if __name__ == "__main__":
    train(parse_args())
