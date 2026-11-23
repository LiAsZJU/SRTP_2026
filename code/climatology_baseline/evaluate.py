"""Score any new baseline using the frozen native-coordinate Argo evaluator."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import cast

import numpy as np
import torch
from dataset_v2.evaluate_checkpoint import Metrics, eligible_validation_platforms, load_profiles
from dataset_v2.grid_sampler import GridSampler
from eof_ridge.model import EOFRidge
from residual_cnn.model import ResidualCNN
from torch import nn
from train_swin_physics import model_input
from unet_baseline.model import UNetReconstructor
from vit_baseline.model import ViTReconstructor

from climatology_baseline.model import ClimatologyBaseline
from climatology_baseline.runtime import (
    JsonObject,
    configure_cuda,
    load_v2_source,
    make_objective,
    write_json,
)


def build_model(config: JsonObject) -> nn.Module:
    """Instantiate the architecture recorded in an immutable baseline checkpoint."""
    architecture = config["architecture"]
    if architecture == "climatology_baseline":
        return ClimatologyBaseline()
    if architecture == "eof_ridge":
        return EOFRidge(config["modes"])
    if architecture == "residual_cnn":
        return ResidualCNN(config["width"], config["blocks"])
    if architecture == "unet_baseline":
        return UNetReconstructor(config["width"])
    if architecture == "vit_baseline":
        return ViTReconstructor(config["embedding_dim"], config["blocks"], config["heads"], config["dropout"])
    raise ValueError(f"Unsupported baseline architecture: {architecture}")


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> JsonObject:
    """Evaluate physical CT/SA RMSE, bias and pooled R-squared on Argo E.

    The implementation reuses the existing Metrics accumulator, platform
    identities, background conversion and native GridSampler. Test reports
    include all E and the subset unseen during validation model selection.
    """
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["schema"] != "ocean_baseline_checkpoint_v1":
        raise ValueError("Use the original evaluator for historical Swin checkpoints")
    device = configure_cuda(config["seed"], args.cpu_threads)
    source, _, _ = load_v2_source(args.dataset_dir)
    manifest_hash = hashlib.sha256((args.dataset_dir / "training_manifest.json").read_bytes()).hexdigest()
    if manifest_hash != config["dataset_manifest_sha256"]:
        raise ValueError("Checkpoint and dataset manifest do not match")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    objective = make_objective(source, device, config)
    mean = cast(torch.Tensor, objective.mean)[0, :, :, 0, 0]
    std = cast(torch.Tensor, objective.std)[0, :, :, 0, 0]
    sampler = GridSampler(Path(next(iter(source.records.values()))["background"]), device)
    validation_platforms = sorted(eligible_validation_platforms(source))
    report: JsonObject = {
        "checkpoint": str(args.checkpoint.resolve()),
        "architecture": config["architecture"],
        "seed": config["seed"],
        "epoch": checkpoint["epoch"],
        "dataset": str(args.dataset_dir.resolve()),
        "manifest_sha256": manifest_hash,
        "device": "cuda",
        "metrics": {},
    }
    for split in args.splits:
        months = sorted(month for month, row in source.records.items() if row["split"] == split)
        if not months:
            raise ValueError(f"No months for split {split}")
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
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(config["amp"])):
                grid = cast(torch.Tensor, model(model_input(sample, device)))
            normalized = sampler(grid.float(), torch.from_numpy(sample["profile_position"]).to(device))
            physical = torch.from_numpy(sample["profile_background"]).to(device) + normalized * std + mean
            errors = normalized.cpu().numpy() - sample["profile_values"]
            for cohort, accumulator in accumulators.items():
                accumulator.add(
                    sample["profile_absolute"],
                    physical.cpu().numpy(),
                    errors,
                    sample["profile_mask"].astype(bool),
                    cohorts[cohort],
                    platforms,
                )
        report["metrics"][split] = {
            "months": months,
            "operators": {
                "native_coordinates": {name: accumulator.summary() for name, accumulator in accumulators.items()}
            },
        }
        print(f"completed split={split} months={len(months)}", flush=True)
    return report


def main() -> None:
    """Write a fresh evaluation report; this command never fits a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--splits", nargs="+", choices=("validation", "test_locked", "forward_test"), required=True)
    args = parser.parse_args()
    write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
