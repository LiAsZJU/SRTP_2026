"""Summarize already completed seed reports without loading data or models."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def summarize(run_dirs: list[Path]) -> dict[str, Any]:
    """Compute means and sample standard deviations for matching experiments.

    A missing or failed seed is an error rather than silently being dropped.
    Statistical references are explicitly marked as nonstochastic controls.
    """
    configs = [json.loads((directory / "config.json").read_text(encoding="utf-8")) for directory in run_dirs]
    reports = [json.loads((directory / "test_metrics.json").read_text(encoding="utf-8")) for directory in run_dirs]
    seeds = [config["seed"] for config in configs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct")
    first = configs[0]
    compared = {
        key: value
        for key, value in first.items()
        if key not in {"seed", "run_dir", "fit_seconds", "retained_training_variance_fraction"}
    }
    for config in configs[1:]:
        candidate = {
            key: value
            for key, value in config.items()
            if key not in {"seed", "run_dir", "fit_seconds", "retained_training_variance_fraction"}
        }
        if candidate != compared:
            raise ValueError("Only the random seed may differ across repeated experiments")
    summary: dict[str, Any] = {
        "architecture": first["architecture"],
        "seeds": seeds,
        "runs": [str(directory.resolve()) for directory in run_dirs],
        "stochastic_model": first["architecture"] in {"residual_cnn", "unet_baseline", "vit_baseline"},
        "std_definition": "sample standard deviation (n-1); null for a single run",
        "manifest_sha256": first["dataset_manifest_sha256"],
        "metrics": {},
    }
    for split in ("test_locked", "forward_test"):
        summary["metrics"][split] = {}
        for cohort in ("all_E", "E_platforms_unseen_in_validation"):
            values = [report["metrics"][split]["operators"]["native_coordinates"][cohort] for report in reports]
            result: dict[str, Any] = {}
            for variable in ("CT", "SA"):
                if len({item[variable]["valid_points"] for item in values}) != 1:
                    raise ValueError("Seed runs have different evaluation support")
                result[variable] = {"valid_points": values[0][variable]["valid_points"]}
                for metric in ("rmse", "bias", "r2"):
                    numbers = [item[variable][metric] for item in values]
                    if any(value is None for value in numbers):
                        result[variable][metric] = {"mean": None, "std": None, "values": numbers}
                    else:
                        result[variable][metric] = {
                            "mean": statistics.mean(numbers),
                            "std": statistics.stdev(numbers) if len(numbers) > 1 else None,
                            "values": numbers,
                        }
            summary["metrics"][split][cohort] = result
    return summary


def main() -> None:
    """Write a new summary after all requested seed runs have completed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(summarize(args.runs), indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
