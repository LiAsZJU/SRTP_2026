"""Group compatible experiments and compute descriptive seed statistics."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from training_visualization.records import (
    COHORTS,
    DETERMINISTIC_MODELS,
    SPLITS,
    VARIABLES,
    RunRecord,
    TestRow,
    test_rows,
)


@dataclass(frozen=True)
class SummaryRow:
    """Across-run statistics with matching reported test months and support counts."""

    group: str
    model: str
    split: str
    cohort: str
    variable: str
    tested_runs: int
    expected_runs: int
    seeds: tuple[int | None, ...]
    missing_seeds: tuple[int, ...]
    planned_epochs: int
    reference: bool
    valid_points: int
    profiles: int
    platforms: int
    rmse_mean: float | None
    rmse_std: float | None
    bias_mean: float | None
    bias_std: float | None
    r2_mean: float | None
    r2_std: float | None


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    """Return a finite-value mean and sample SD; one run has no estimated SD."""
    observed = [value for value in values if value is not None]
    return (
        (statistics.mean(observed), statistics.stdev(observed) if len(observed) > 1 else None)
        if observed
        else (None, None)
    )


def group_runs(runs: list[RunRecord]) -> dict[str, list[RunRecord]]:
    """Group by architecture and scientific hyperparameters, preserving source provenance."""
    if len({run.manifest_sha256 for run in runs}) != 1:
        raise ValueError("Selected runs use different dataset manifests; compare one protocol at a time")
    groups: dict[str, list[RunRecord]] = defaultdict(list)
    for run in runs:
        groups[run.group].append(run)
    return dict(sorted(groups.items()))


def summarize(
    runs: list[RunRecord], expected_seeds: tuple[int, ...], *, allow_partial: bool
) -> tuple[list[SummaryRow], list[TestRow], list[str]]:
    """Validate support and aggregate each model, split, cohort and variable.

    Error bars describe seed variability; they are not confidence intervals.
    Historical references and deterministic models are singleton controls.
    """
    if len(set(expected_seeds)) != len(expected_seeds) or not expected_seeds:
        raise ValueError("Expected seeds must be nonempty and distinct")
    groups = group_runs(runs)
    raw_rows: list[TestRow] = []
    summaries = []
    notes = [note for run in runs for note in run.notes]
    support: dict[tuple[str, str, str], tuple[tuple[str, ...], int, int, int]] = {}
    for group, members in groups.items():
        first = members[0]
        seeds = [run.seed for run in members]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate seeds in {group}; select one run per seed")
        completed = [run for run in members if run.report is not None]
        singleton = first.model in DETERMINISTIC_MODELS or first.is_reference
        if singleton and len(members) > 1:
            raise ValueError(f"Select one deterministic/reference run for {group}")
        tested_seeds = {run.seed for run in completed}
        missing = () if singleton else tuple(seed for seed in expected_seeds if seed not in tested_seeds)
        unexpected = set(seeds) - set(expected_seeds) if not singleton else set()
        if unexpected:
            raise ValueError(f"Unexpected seeds in {group}: {unexpected}; set --expected-seeds explicitly")
        expected_count = 1 if singleton else len(expected_seeds)
        if len(completed) != expected_count:
            message = (
                f"{first.display_name} [{group}]: tested {len(completed)}/{expected_count}; missing seeds {missing}"
            )
            if not allow_partial:
                raise ValueError(message + "; wait for completion or use --allow-partial for a labelled preview")
            notes.append(message)
        rows = [row for run in completed for row in test_rows(run)]
        raw_rows.extend(rows)
        for row in rows:
            key = (row.split, row.cohort, row.variable)
            identity = (row.months, row.valid_points, row.profiles, row.platforms)
            if key in support and support[key] != identity:
                raise ValueError(f"Different test months or observation support for {key}; do not average these scores")
            support[key] = identity
        if not rows:
            continue
        for split in SPLITS:
            for cohort in COHORTS:
                for variable in VARIABLES:
                    selected = [
                        row for row in rows if (row.split, row.cohort, row.variable) == (split, cohort, variable)
                    ]
                    rmse, rmse_std = mean_std([row.rmse for row in selected])
                    bias, bias_std = mean_std([row.bias for row in selected])
                    r2, r2_std = mean_std([row.r2 for row in selected])
                    summaries.append(
                        SummaryRow(
                            group,
                            first.model,
                            split,
                            cohort,
                            variable,
                            len(completed),
                            expected_count,
                            tuple(run.seed for run in completed),
                            missing,
                            first.planned_epochs,
                            first.is_reference,
                            selected[0].valid_points,
                            selected[0].profiles,
                            selected[0].platforms,
                            rmse,
                            rmse_std,
                            bias,
                            bias_std,
                            r2,
                            r2_std,
                        )
                    )
    budgets = {run.planned_epochs for run in runs if run.model not in DETERMINISTIC_MODELS}
    if len(budgets) > 1:
        notes.append(
            "Different neural training budgets are included. This is a descriptive comparison, not a controlled architecture ranking."
        )
    notes.append(
        "Pooled R-squared combines valid profiles and depths. Depth-wise skill cannot be recovered from these aggregate reports."
    )
    notes.append(
        "Test months and reported support counts are checked. Aggregate JSON cannot prove per-observation identity matching."
    )
    notes.append(
        "Seed means describe separate trained models, not the performance of an ensemble of their predictions."
    )
    notes.append(
        "Argo E can overlap upstream CORA analysis observations; this is not a fully independent external evaluation."
    )
    return summaries, raw_rows, notes


def run_inventory(runs: list[RunRecord]) -> list[dict[str, Any]]:
    """Describe completed work and timing without treating missing metrics as zero."""
    rows = []
    for run in runs:
        seconds = [float(row["seconds"]) for row in run.history if row.get("seconds") is not None]
        warm = seconds[5:]
        rows.append(
            {
                "run": str(run.path),
                "group": run.group,
                "model": run.model,
                "seed": run.seed,
                "reference": run.is_reference,
                "planned_epochs": run.planned_epochs,
                "completed_epochs": run.completed_epochs,
                "best_epoch": run.best_epoch,
                "test_report_available": run.report is not None,
                "parameters": run.config.get("parameters"),
                "train_validation_seconds_sum": sum(seconds) if seconds else None,
                "warm_epoch_seconds_median": statistics.median(warm) if warm else None,
                "device": run.config.get("device_resolved", run.config.get("device")),
                "gpu": run.config.get("gpu"),
                "manifest_sha256": run.manifest_sha256,
            }
        )
    return rows


def summary_dicts(rows: list[SummaryRow]) -> list[dict[str, Any]]:
    """Convert typed summary records to exportable rows."""
    return [asdict(row) for row in rows]
