"""Publication-grade Matplotlib figures for depth/spatial diagnostics and CT/SA slices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, TwoSlopeNorm

from spatial_depth_diagnostics.metrics import SpatialBinAccumulator
from spatial_depth_diagnostics.slicing import OceanSlice

PALETTE: tuple[str, ...] = (
    "#c0392b",
    "#2980b9",
    "#27ae60",
    "#e67e22",
    "#8e44ad",
    "#16a085",
    "#7f8c8d",
)


def _save_figure(fig: plt.Figure, stem: Path, formats: tuple[str, ...]) -> list[str]:
    """Save figure in all requested formats and return relative filenames."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for fmt in formats:
        out_path = stem.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        saved.append(out_path.name)
    plt.close(fig)
    return saved


def _axis_edges(centers: np.ndarray) -> np.ndarray:
    """Compute monotonic cell edges from 1D (possibly non-uniform) coordinate centers."""
    c = np.asarray(centers, dtype=np.float64)
    if c.size < 2:
        return np.array([c[0] - 0.5, c[0] + 0.5], dtype=np.float64)
    mid = 0.5 * (c[:-1] + c[1:])
    first = c[0] - 0.5 * (c[1] - c[0])
    last = c[-1] + 0.5 * (c[-1] - c[-2])
    return np.concatenate(([first], mid, [last]))


def _robust_symmetric_limit(values: np.ndarray, floor: float = 1e-4) -> float:
    """Return a finite positive symmetric color limit using the 98th percentile of |values|."""
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return floor
    limit = float(np.percentile(finite, 98.0))
    return max(limit, floor)


def _robust_range(values: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> tuple[float, float]:
    """Return robust finite (vmin, vmax) bounds for absolute physical fields."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, low_q))
    vmax = float(np.percentile(finite, high_q))
    if vmax <= vmin:
        return vmin - 0.5, vmax + 0.5
    return vmin, vmax


def _masked_cmap(name: str, bad_color: str = "#dcdcdc") -> Colormap:
    """Copy a named Matplotlib colormap and render NaN/land cells in neutral gray."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(bad_color)
    return cmap


def plot_depth_error_curves(
    depth_rows: list[dict[str, Any]],
    split: str,
    cohort: str,
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[str]:
    """Plot CT and SA depth-wise RMSE, Bias, and R-squared profiles (5-500 m)."""
    filtered = [row for row in depth_rows if row["split"] == split and row["cohort"] == cohort]
    if not filtered:
        return []
    models = list(dict.fromkeys(str(row["model"]) for row in filtered))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.8), sharey=True)
    var_specs = (("CT", "deg C"), ("SA", "g/kg"))
    metric_specs = (
        ("rmse", "RMSE"),
        ("bias", "Bias (Pred - Obs)"),
        ("r2", "R-squared"),
    )

    for row_idx, (variable, unit) in enumerate(var_specs):
        for col_idx, (metric_key, metric_label) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            for m_idx, model in enumerate(models):
                pts = sorted(
                    (r for r in filtered if r["model"] == model and r["variable"] == variable),
                    key=lambda item: float(item["depth_m"]),
                )
                depths = np.array([float(p["depth_m"]) for p in pts], dtype=np.float64)
                vals = np.array(
                    [np.nan if p[metric_key] is None else float(p[metric_key]) for p in pts],
                    dtype=np.float64,
                )
                color = PALETTE[m_idx % len(PALETTE)]
                ax.plot(vals, depths, label=model, color=color, linewidth=1.8)
            if metric_key == "bias":
                ax.axvline(0.0, color="#333333", linestyle="--", linewidth=0.9, alpha=0.7)
            ax.grid(True, linestyle=":", alpha=0.45)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            xlabel = f"{variable} {metric_label} [{unit}]" if metric_key != "r2" else f"{variable} {metric_label}"
            ax.set_xlabel(xlabel)
            if col_idx == 0:
                ax.set_ylabel("Depth [m]")
            if row_idx == 0:
                ax.set_title(metric_label)

    axes[0, 0].invert_yaxis()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 6),
            frameon=False,
            bbox_to_anchor=(0.5, 0.96),
        )
    fig.suptitle(f"Depth-wise Argo E Diagnostics | {split} | {cohort}", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return _save_figure(fig, output_dir / f"depth_error_{split}_{cohort}", formats)


def plot_spatial_error_maps(
    spatial_acc: SpatialBinAccumulator,
    model: str,
    split: str,
    cohort: str,
    band_name: str,
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[str]:
    """Plot 2x2 geographic maps of binned Argo E CT/SA RMSE and Bias."""
    maps = spatial_acc.grid_maps(band_name)
    total_profiles = int(maps["profiles"].sum())
    if total_profiles == 0:
        return []

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 6.8), sharex=True, sharey=True)
    panels = (
        (0, 0, maps["ct_rmse"], "CT RMSE [deg C]", "YlOrRd", False),
        (0, 1, maps["ct_bias"], "CT Bias (Pred - Obs) [deg C]", "RdBu_r", True),
        (1, 0, maps["sa_rmse"], "SA RMSE [g/kg]", "YlOrRd", False),
        (1, 1, maps["sa_bias"], "SA Bias (Pred - Obs) [g/kg]", "RdBu_r", True),
    )

    for r_idx, c_idx, data, title, cmap_name, symmetric in panels:
        ax = axes[r_idx, c_idx]
        cmap = _masked_cmap(cmap_name)
        if symmetric:
            lim = _robust_symmetric_limit(data, floor=0.02 if "SA" in title else 0.1)
            mesh = ax.pcolormesh(
                spatial_acc.lon_edges,
                spatial_acc.lat_edges,
                data,
                cmap=cmap,
                norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                shading="flat",
            )
        else:
            _, vmax = _robust_range(data, low_q=0.0, high_q=98.0)
            mesh = ax.pcolormesh(
                spatial_acc.lon_edges,
                spatial_acc.lat_edges,
                data,
                cmap=cmap,
                vmin=0.0,
                vmax=max(vmax, 1e-4),
                shading="flat",
            )
        ax.set_title(title)
        ax.set_facecolor("#ededed")
        ax.grid(True, linestyle=":", alpha=0.4)
        if r_idx == 1:
            ax.set_xlabel("Longitude [deg E]")
        if c_idx == 0:
            ax.set_ylabel("Latitude [deg N]")
        fig.colorbar(mesh, ax=ax, shrink=0.88, pad=0.02)

    fig.suptitle(
        f"Spatial Argo E Error ({spatial_acc.bin_deg:g} deg bins) | {model} | {split} | {cohort} | {band_name} "
        f"(profiles={total_profiles})",
        y=0.995,
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    safe_model = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model)
    return _save_figure(fig, output_dir / f"spatial_error_{safe_model}_{split}_{cohort}_{band_name}", formats)


def plot_representative_profiles(
    profiles: list[dict[str, Any]],
    depths: np.ndarray,
    split: str,
    cohort: str,
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[str]:
    """Plot vertical CT and SA comparisons for representative held-out Argo E profiles."""
    if not profiles:
        return []
    n_prof = len(profiles)
    fig, axes = plt.subplots(2, n_prof, figsize=(3.1 * n_prof + 1.2, 7.6), sharey=True, squeeze=False)
    model_names = list(profiles[0]["predictions"].keys())

    for col_idx, prof in enumerate(profiles):
        obs_ct = np.asarray(prof["observed_ct"], dtype=np.float64)
        obs_sa = np.asarray(prof["observed_sa"], dtype=np.float64)
        bg_ct = np.asarray(prof["background_ct"], dtype=np.float64)
        bg_sa = np.asarray(prof["background_sa"], dtype=np.float64)

        ax_ct = axes[0, col_idx]
        ax_sa = axes[1, col_idx]

        ax_ct.plot(obs_ct, depths, color="#111111", linewidth=2.0, label="Argo E Obs")
        ax_ct.plot(bg_ct, depths, color="#7f8c8d", linestyle="--", linewidth=1.3, label="Climatology")
        ax_sa.plot(obs_sa, depths, color="#111111", linewidth=2.0, label="Argo E Obs")
        ax_sa.plot(bg_sa, depths, color="#7f8c8d", linestyle="--", linewidth=1.3, label="Climatology")

        for m_idx, model in enumerate(model_names):
            color = PALETTE[m_idx % len(PALETTE)]
            pred_ct = np.asarray(prof["predictions"][model]["CT"], dtype=np.float64)
            pred_sa = np.asarray(prof["predictions"][model]["SA"], dtype=np.float64)
            ax_ct.plot(pred_ct, depths, color=color, linewidth=1.5, label=model)
            ax_sa.plot(pred_sa, depths, color=color, linewidth=1.5, label=model)

        lat_val = float(prof["latitude"])
        lon_val = float(prof["longitude"])
        pct = float(prof["percentile"])
        ax_ct.set_title(
            f"P{pct:.0f} | {prof['month']} #{prof['platform']}\n({lat_val:+.1f}N, {lon_val:.1f}E)",
            fontsize=9.5,
        )
        ax_ct.set_xlabel("CT [deg C]")
        ax_sa.set_xlabel("SA [g/kg]")
        for ax in (ax_ct, ax_sa):
            ax.grid(True, linestyle=":", alpha=0.45)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        if col_idx == 0:
            ax_ct.set_ylabel("Depth [m]")
            ax_sa.set_ylabel("Depth [m]")

    axes[0, 0].invert_yaxis()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(labels), 6),
        frameon=False,
        bbox_to_anchor=(0.5, 0.96),
    )
    fig.suptitle(f"Representative Argo E Vertical Profiles | {split} | {cohort}", y=0.995, fontsize=12.5)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return _save_figure(fig, output_dir / f"representative_profiles_{split}_{cohort}", formats)


def plot_ocean_slice(
    ocean_slice: OceanSlice,
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[str]:
    """Plot 2D horizontal depth map or vertical cross-section of CT and SA (with optional CORA/anomaly panels)."""
    columns: list[tuple[str, np.ndarray, np.ndarray, bool]] = [
        (ocean_slice.source_label, ocean_slice.ct, ocean_slice.sa, False)
    ]
    if "CT_cora" in ocean_slice.extras and "SA_cora" in ocean_slice.extras:
        columns.append(("CORA Reference", ocean_slice.extras["CT_cora"], ocean_slice.extras["SA_cora"], False))
    if "CT_diff_cora" in ocean_slice.extras and "SA_diff_cora" in ocean_slice.extras:
        columns.append(
            ("Diff (Source - CORA)", ocean_slice.extras["CT_diff_cora"], ocean_slice.extras["SA_diff_cora"], True)
        )
    if "CT_anomaly" in ocean_slice.extras and "SA_anomaly" in ocean_slice.extras:
        columns.append(
            ("Anomaly (Source - Clim)", ocean_slice.extras["CT_anomaly"], ocean_slice.extras["SA_anomaly"], True)
        )

    n_cols = len(columns)
    is_vertical = ocean_slice.kind in ("zonal", "meridional")
    panel_w = 5.4 if ocean_slice.kind == "horizontal" else 5.0
    panel_h = 3.2 if ocean_slice.kind == "horizontal" else 3.6
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(panel_w * n_cols + 0.8, panel_h * 2 + 1.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    x_edges = _axis_edges(ocean_slice.x_values)
    y_edges = _axis_edges(ocean_slice.y_values)

    # Share absolute CT/SA color bounds between source and CORA reference when both are plotted
    abs_ct_arrays = [ct for _, ct, _, is_diff in columns if not is_diff]
    abs_sa_arrays = [sa for _, _, sa, is_diff in columns if not is_diff]
    ct_vmin, ct_vmax = _robust_range(np.concatenate([a[np.isfinite(a)] for a in abs_ct_arrays]))
    sa_vmin, sa_vmax = _robust_range(np.concatenate([a[np.isfinite(a)] for a in abs_sa_arrays]))

    for col_idx, (col_title, ct_field, sa_field, is_diff) in enumerate(columns):
        for row_idx, (var_name, unit, field, abs_cmap, abs_bounds) in enumerate(
            (
                ("CT", "deg C", ct_field, "RdYlBu_r", (ct_vmin, ct_vmax)),
                ("SA", "g/kg", sa_field, "viridis", (sa_vmin, sa_vmax)),
            )
        ):
            ax = axes[row_idx, col_idx]
            if is_diff:
                cmap = _masked_cmap("RdBu_r")
                lim = _robust_symmetric_limit(field, floor=0.05 if var_name == "SA" else 0.25)
                mesh = ax.pcolormesh(
                    x_edges,
                    y_edges,
                    field,
                    cmap=cmap,
                    norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                    shading="flat",
                )
            else:
                cmap = _masked_cmap(abs_cmap)
                mesh = ax.pcolormesh(
                    x_edges,
                    y_edges,
                    field,
                    cmap=cmap,
                    vmin=abs_bounds[0],
                    vmax=abs_bounds[1],
                    shading="flat",
                )
                if is_vertical and np.isfinite(field).sum() > 20:
                    levels = np.linspace(abs_bounds[0], abs_bounds[1], 7)
                    if np.all(np.diff(levels) > 0):
                        cs = ax.contour(
                            ocean_slice.x_values,
                            ocean_slice.y_values,
                            field,
                            levels=levels,
                            colors="#222222",
                            linewidths=0.6,
                            alpha=0.55,
                        )
                        ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

            ax.set_title(f"{col_title} | {var_name} [{unit}]", fontsize=10.5)
            ax.grid(True, linestyle=":", alpha=0.35)
            if row_idx == 1:
                ax.set_xlabel(f"{ocean_slice.x_name.title()} [{ocean_slice.x_unit}]")
            if col_idx == 0:
                ax.set_ylabel(f"{ocean_slice.y_name.title()} [{ocean_slice.y_unit}]")
            fig.colorbar(mesh, ax=ax, shrink=0.88, pad=0.02)

    if is_vertical:
        axes[0, 0].invert_yaxis()

    fc = ocean_slice.fixed_coord
    coord_desc = f"{fc.axis_name}={fc.matched:.2f} {fc.unit}"
    if abs(fc.matched - fc.requested) > 1e-3:
        coord_desc += f" (req {fc.requested:g})"
    fig.suptitle(
        f"Ocean {ocean_slice.kind.title()} Slice | {ocean_slice.source_label} | {ocean_slice.month} | {coord_desc}",
        y=0.995,
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return _save_figure(fig, output_dir / ocean_slice.slug, formats)
