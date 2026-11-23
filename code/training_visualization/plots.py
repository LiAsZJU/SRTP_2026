"""Export static scientific figures with a CPU-only Matplotlib Agg canvas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from training_visualization.analysis import SummaryRow, group_runs, run_inventory
from training_visualization.records import COHORTS, DETERMINISTIC_MODELS, MODEL_NAMES, SPLITS, RunRecord, TestRow

COLORS = {
    "climatology_baseline": "#7f7f7f",
    "eof_ridge": "#9467bd",
    "residual_cnn": "#1f77b4",
    "unet_baseline": "#2ca02c",
    "vit_baseline": "#ff7f0e",
    "swin": "#d62728",
}
SEED_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
SPLIT_NAMES = {"test_locked": "2024 main test", "forward_test": "2025 Jan-Jun supplementary test"}


def _save(figure: Figure, output: Path, stem: str, formats: tuple[str, ...]) -> list[str]:
    paths = []
    for extension in formats:
        path = output / f"{stem}.{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(str(path.relative_to(output)))
    figure.clear()
    return paths


def _style(axis: Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)


def learning_curves(runs: list[RunRecord], output: Path, formats: tuple[str, ...], *, partial: bool) -> list[str]:
    """Draw individual seed curves and mark validation-selected epochs."""
    files = []
    fields = ("train_total", "validation_score", "validation_CT_RMSE_C", "validation_SA_RMSE_g_kg")
    titles = (
        "Total training objective",
        "Normalized validation score",
        "Validation CT RMSE [deg C]",
        "Validation SA RMSE [g/kg]",
    )
    for group, members in group_runs(runs).items():
        figure = Figure(figsize=(12, 8), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(2, 2)
        for index, run in enumerate(sorted(members, key=lambda member: member.seed or 0)):
            for axis, field, title in zip(axes.flat, fields, titles, strict=True):
                rows = [row for row in run.history if row.get(field) is not None]
                if rows:
                    color = SEED_COLORS[index % len(SEED_COLORS)]
                    axis.plot(
                        [row["epoch"] for row in rows],
                        [row[field] for row in rows],
                        color=color,
                        linewidth=1.25,
                        label=f"seed {run.seed}",
                        marker="o" if len(rows) == 1 else None,
                    )
                    best = next((row for row in rows if row["epoch"] == run.best_epoch), None)
                    if best is not None:
                        axis.scatter([best["epoch"]], [best[field]], color=color, marker="*", s=90, zorder=4)
                axis.set_title(title, fontsize=11)
                axis.set_xlabel("Logged epoch (0 = packaged climatology)")
                _style(axis)
        for axis in axes.flat:
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(handles, labels, fontsize=8)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No iterative training" if members[0].model in DETERMINISTIC_MODELS else "No logged values yet",
                    ha="center",
                    transform=axis.transAxes,
                )
        marker = " | PARTIAL SNAPSHOT" if partial else ""
        figure.suptitle(f"{members[0].display_name} | {group} | stars: validation-selected epoch{marker}", fontsize=13)
        files.extend(_save(figure, output, f"learning_curves/{group}", formats))
    return files


def _group_label(row: SummaryRow, duplicates: bool) -> str:
    budget = (
        "fixed"
        if row.model == "climatology_baseline"
        else "one fit"
        if row.model == "eof_ridge"
        else f"{row.planned_epochs} ep"
    )
    suffix = " ref" if row.reference else ""
    variant = " #" + row.group.removesuffix("_reference").rsplit("_", 1)[-1][:6] if duplicates else ""
    return (
        f"{MODEL_NAMES.get(row.model, row.model)}{variant}\n{budget}{suffix}, n={row.tested_runs}/{row.expected_runs}"
    )


def comparison_plots(
    summaries: list[SummaryRow],
    observations: list[TestRow],
    output: Path,
    formats: tuple[str, ...],
    *,
    partial: bool,
) -> list[str]:
    """Plot RMSE, R-squared and signed bias separately for each Argo cohort.

    Each point is one seed's pooled score. Error bars are sample standard
    deviations, never confidence intervals or standard errors.
    """
    files = []
    metrics = (("rmse", "RMSE"), ("r2", "R-squared"), ("bias", "Signed bias"))
    for cohort in COHORTS:
        for metric, title in metrics:
            figure = Figure(figsize=(14, 8), layout="constrained")
            FigureCanvasAgg(figure)
            axes = figure.subplots(2, 2)
            for column, split in enumerate(SPLITS):
                for row_index, variable in enumerate(("CT", "SA")):
                    axis = axes[row_index, column]
                    selected = [
                        item
                        for item in summaries
                        if (item.split, item.cohort, item.variable) == (split, cohort, variable)
                    ]
                    labels = []
                    for position, item in enumerate(selected):
                        mean = getattr(item, f"{metric}_mean")
                        deviation = getattr(item, f"{metric}_std")
                        labels.append(_group_label(item, sum(other.model == item.model for other in selected) > 1))
                        if mean is None:
                            continue
                        color = COLORS.get(item.model, "#333333")
                        if metric == "r2":
                            axis.scatter([position], [mean], s=60, color=color, zorder=3)
                        else:
                            axis.bar(position, mean, width=0.62, color=color, alpha=0.72)
                        if deviation is not None:
                            axis.errorbar(
                                position, mean, yerr=deviation, color="#222222", capsize=4, fmt="none", zorder=4
                            )
                        seeds = [
                            getattr(value, metric)
                            for value in observations
                            if (value.group, value.split, value.cohort, value.variable)
                            == (item.group, split, cohort, variable)
                        ]
                        values = [value for value in seeds if value is not None]
                        jitter = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else np.zeros(len(values))
                        axis.scatter(
                            position + jitter,
                            values,
                            s=22,
                            facecolor="white",
                            edgecolor="#222222",
                            linewidth=0.8,
                            zorder=5,
                        )
                    axis.set_xticks(range(len(selected)), labels, fontsize=8)
                    unit = "" if metric == "r2" else (" [deg C]" if variable == "CT" else " [g/kg]")
                    axis.set_ylabel(variable + " " + title + unit)
                    axis.set_title(SPLIT_NAMES[split], fontsize=11)
                    if metric == "rmse":
                        axis.set_ylim(bottom=0)
                    if metric == "bias":
                        axis.axhline(0, color="#222222", linewidth=0.8)
                    if not selected:
                        axis.text(0.5, 0.5, "No completed test reports", ha="center", transform=axis.transAxes)
                    _style(axis)
            cohort_title = "All held-out Argo E" if cohort == "all_E" else "Argo E platforms unseen during validation"
            state = "PARTIAL SNAPSHOT | " if partial else ""
            figure.suptitle(
                f"{state}{cohort_title} | {title}\nDots: individual seeds; error bars: sample SD (n-1); n=1 has no estimated SD",
                fontsize=12,
            )
            files.extend(_save(figure, output, f"test_{metric}_{cohort}", formats))
    return files


def completion_and_timing(runs: list[RunRecord], output: Path, formats: tuple[str, ...], *, partial: bool) -> list[str]:
    """Show actual epoch completion and warm median epoch time, excluding the first five epochs."""
    inventory = run_inventory(runs)
    figure = Figure(figsize=(13, max(5.0, len(runs) * 0.38)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    labels = [f"{run.display_name} | seed {run.seed}" for run in runs]
    for position, (run, record) in enumerate(zip(runs, inventory, strict=True)):
        color = COLORS.get(run.model, "#333333")
        if run.model in DETERMINISTIC_MODELS:
            axes[0].text(
                0.02,
                position,
                "No iterative training" if run.model == "climatology_baseline" else "One closed-form fit",
                transform=axes[0].get_yaxis_transform(),
                va="center",
                fontsize=9,
            )
        else:
            axes[0].barh(position, run.planned_epochs, color="#eeeeee", height=0.7)
            axes[0].barh(position, run.completed_epochs, color=color, height=0.5)
        median: Any = record["warm_epoch_seconds_median"]
        if median is not None:
            axes[1].barh(position, median, color=color, height=0.5)
    for axis in axes:
        axis.set_yticks(range(len(runs)), labels, fontsize=8)
        axis.invert_yaxis()
        _style(axis)
    axes[0].set_xlabel("Actual logged epoch / configured maximum (background)")
    axes[1].set_xlabel("Median training + validation seconds / epoch, excluding first 5")
    state = "PARTIAL SNAPSHOT | " if partial else ""
    figure.suptitle(
        state
        + "Training budget and observed epoch timing\nTiming excludes checkpoint writes and final tests; concurrent workloads affect speed",
        fontsize=12,
    )
    return _save(figure, output, "completion_and_epoch_timing", formats)
