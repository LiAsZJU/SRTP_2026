"""CLI for read-only spatial/depth diagnostics and 2D ocean CT/SA slice generation."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from spatial_depth_diagnostics.loader import (
    LoadedModel,
    ModelSpec,
    load_frozen_model,
    open_v2_dataset,
    parse_model_spec,
    resolve_device,
    resolve_months,
    run_month_inference,
    target_norm_stats,
)
from spatial_depth_diagnostics.metrics import (
    DEFAULT_DEPTH_BANDS,
    DepthMetricsAccumulator,
    RepresentativeProfileCollector,
    SpatialBinAccumulator,
)
from spatial_depth_diagnostics.plots import (
    plot_depth_error_curves,
    plot_ocean_slice,
    plot_representative_profiles,
    plot_spatial_error_maps,
)
from spatial_depth_diagnostics.slicing import (
    OceanSlice,
    export_slice_netcdf,
    export_slice_npz,
    extract_horizontal_slice,
    extract_meridional_section,
    extract_zonal_section,
    slice_summary_dict,
)

CORA_OVERLAP_NOTE = (
    "Held-out Argo E profiles can overlap upstream CORA analysis observations; "
    "scores are held-out Argo evaluations rather than fully independent external validation."
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write diagnostic tables to UTF-8 CSV without rounding underlying numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _cohort_masks(
    roles: np.ndarray,
    platforms: np.ndarray,
    validation_platforms: set[str],
    split: str,
    requested_cohorts: list[str] | None,
) -> dict[str, np.ndarray]:
    """Build boolean Argo E cohort masks matching the frozen v2 evaluation protocol."""
    all_e = roles == 2
    available: dict[str, np.ndarray] = {"all_E": all_e}
    if split != "validation":
        unseen = all_e & ~np.isin(platforms, sorted(validation_platforms))
        available["E_platforms_unseen_in_validation"] = unseen
    if requested_cohorts:
        return {name: available[name] for name in requested_cohorts if name in available}
    return available


def run_diagnose(args: argparse.Namespace) -> Path:
    """Execute depth-wise, spatial-bin, and representative-profile diagnostics."""
    from dataset_v2.grid_sampler import GridSampler

    output_dir: Path = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    specs: list[ModelSpec] = [parse_model_spec(raw) for raw in args.model]
    if len({spec.label for spec in specs}) != len(specs):
        raise ValueError("Duplicate model labels provided to --model")

    device = resolve_device(args.device, args.cpu_threads)
    source, manifest_sha256, axes, validation_platforms = open_v2_dataset(args.dataset_dir)
    norm_mean, norm_std = target_norm_stats(source)
    first_bg = Path(next(iter(source.records.values()))["background"])
    sampler = GridSampler(first_bg, device)

    loaded_models: list[LoadedModel] = [load_frozen_model(spec, manifest_sha256, device) for spec in specs]
    month_pairs = resolve_months(source, splits=args.splits, explicit_months=args.months)
    split_groups: dict[str, list[str]] = {}
    for month, split in month_pairs:
        split_groups.setdefault(split, []).append(month)

    depth_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    figures: list[str] = []
    profiles_export: dict[str, dict[str, list[dict[str, Any]]]] = {}
    formats = tuple(args.formats)

    output_dir.mkdir(parents=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True)

    for split, months in split_groups.items():
        cohorts = list(
            _cohort_masks(
                np.array([], dtype=np.int64),
                np.array([], dtype=str),
                validation_platforms,
                split,
                args.cohorts,
            ).keys()
        )
        depth_accs = {
            (lm.label, cohort): DepthMetricsAccumulator(axes.depth) for lm in loaded_models for cohort in cohorts
        }
        spatial_accs = {
            (lm.label, cohort): SpatialBinAccumulator(axes.latitude, axes.longitude, axes.depth, bin_deg=args.bin_deg)
            for lm in loaded_models
            for cohort in cohorts
        }
        collectors = {cohort: RepresentativeProfileCollector(args.min_valid_fraction) for cohort in cohorts}

        for month in months:
            for lm in loaded_models:
                inf = run_month_inference(source, month, lm, sampler, norm_mean, norm_std)
                masks = _cohort_masks(
                    inf.profile_role_code,
                    inf.profile_platform,
                    validation_platforms,
                    split,
                    args.cohorts,
                )
                for cohort, mask in masks.items():
                    depth_accs[(lm.label, cohort)].add(
                        inf.profile_observed, inf.profile_predicted, inf.profile_valid, mask
                    )
                    spatial_accs[(lm.label, cohort)].add(
                        inf.profile_observed,
                        inf.profile_predicted,
                        inf.profile_valid,
                        inf.profile_lat,
                        inf.profile_lon,
                        mask,
                    )
                    collectors[cohort].add_month(
                        lm.label,
                        month,
                        inf.profile_observed,
                        inf.profile_predicted,
                        inf.profile_background,
                        inf.profile_valid,
                        inf.profile_lat,
                        inf.profile_lon,
                        inf.profile_time_offset,
                        inf.profile_platform,
                        mask,
                    )

        profiles_export[split] = {}
        model_labels = [lm.label for lm in loaded_models]
        ref_model = args.reference_model or model_labels[0]
        if ref_model not in model_labels:
            raise ValueError(f"--reference-model {ref_model!r} is not in provided models {model_labels}")

        for cohort in cohorts:
            split_cohort_depth_rows: list[dict[str, Any]] = []
            for lm in loaded_models:
                rows_d = depth_accs[(lm.label, cohort)].to_rows(lm.label, split, cohort)
                split_cohort_depth_rows.extend(rows_d)
                depth_rows.extend(rows_d)
                spatial_rows.extend(spatial_accs[(lm.label, cohort)].to_rows(lm.label, split, cohort))
                for band_name, _, _ in DEFAULT_DEPTH_BANDS:
                    if args.spatial_bands and band_name not in args.spatial_bands:
                        continue
                    saved = plot_spatial_error_maps(
                        spatial_accs[(lm.label, cohort)],
                        lm.label,
                        split,
                        cohort,
                        band_name,
                        fig_dir,
                        formats,
                    )
                    figures.extend(f"figures/{name}" for name in saved)

            saved_depth = plot_depth_error_curves(split_cohort_depth_rows, split, cohort, fig_dir, formats)
            figures.extend(f"figures/{name}" for name in saved_depth)

            selected = collectors[cohort].select(ref_model, model_labels, count=args.num_profiles)
            profiles_export[split][cohort] = selected
            saved_prof = plot_representative_profiles(selected, axes.depth, split, cohort, fig_dir, formats)
            figures.extend(f"figures/{name}" for name in saved_prof)

    write_csv(output_dir / "depth_metrics.csv", depth_rows)
    write_csv(output_dir / "spatial_error_bins.csv", spatial_rows)
    (output_dir / "representative_profiles.json").write_text(
        json.dumps({"depths_m": axes.depth.tolist(), "profiles": profiles_export}, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "mode": "diagnose",
        "device": str(device),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "manifest_sha256": manifest_sha256,
        "splits": split_groups,
        "bin_deg": args.bin_deg,
        "models": [
            {
                "label": lm.label,
                "architecture": lm.architecture,
                "checkpoint": str(lm.checkpoint),
                "checkpoint_sha256": lm.checkpoint_sha256,
                "epoch": lm.epoch,
                "seed": lm.seed,
            }
            for lm in loaded_models
        ],
        "notes": [CORA_OVERLAP_NOTE],
        "figures": figures,
    }
    (output_dir / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report_readme(output_dir, "Spatial & Depth Diagnostics Report", summary, figures)
    return output_dir


def run_slice(args: argparse.Namespace) -> Path:
    """Extract and export 2D horizontal/zonal/meridional ocean CT/SA slices (data + figures)."""
    from dataset_v2.grid_sampler import GridSampler

    output_dir: Path = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if not args.depths and not args.zonal_lats and not args.meridional_lons:
        raise ValueError("Specify at least one of --depths, --zonal-lats, or --meridional-lons")

    device = resolve_device(args.device, args.cpu_threads)
    source, manifest_sha256, axes, _ = open_v2_dataset(args.dataset_dir)
    norm_mean, norm_std = target_norm_stats(source)
    first_bg = Path(next(iter(source.records.values()))["background"])
    sampler = GridSampler(first_bg, device)

    sources_to_run: list[tuple[str, LoadedModel | None, Any]] = []
    if args.model:
        specs = [parse_model_spec(raw) for raw in args.model]
        if len({spec.label for spec in specs}) != len(specs):
            raise ValueError("Duplicate model labels provided to --model")
        for spec in specs:
            lm = load_frozen_model(spec, manifest_sha256, device)
            sources_to_run.append((lm.label, lm, None))
    if args.direct_source:
        for direct in args.direct_source:
            sources_to_run.append((direct, None, direct))
    if not sources_to_run:
        raise ValueError("Specify at least one --model or --direct-source (cora / background)")

    month_pairs = resolve_months(source, splits=args.splits, explicit_months=args.months)
    output_dir.mkdir(parents=True)
    fig_dir = output_dir / "figures"
    data_dir = output_dir / "slices"
    fig_dir.mkdir(parents=True)
    if args.export_data:
        data_dir.mkdir(parents=True)

    formats = tuple(args.formats)
    figures: list[str] = []
    slice_records: list[dict[str, Any]] = []

    for month, split in month_pairs:
        for label, lm, direct in sources_to_run:
            inf = run_month_inference(source, month, lm, sampler, norm_mean, norm_std, direct_source=direct)
            include_cora = bool(args.include_cora and direct != "cora")
            include_anomaly = bool(args.include_anomaly and direct != "background")

            slices: list[OceanSlice] = []
            for depth in args.depths:
                slices.append(
                    extract_horizontal_slice(
                        inf.predicted_grid,
                        axes,
                        month,
                        label,
                        depth,
                        cora_grid=inf.cora_grid,
                        background_grid=inf.background_grid,
                        include_cora=include_cora,
                        include_anomaly=include_anomaly,
                    )
                )
            for lat in args.zonal_lats:
                slices.append(
                    extract_zonal_section(
                        inf.predicted_grid,
                        axes,
                        month,
                        label,
                        lat,
                        cora_grid=inf.cora_grid,
                        background_grid=inf.background_grid,
                        include_cora=include_cora,
                        include_anomaly=include_anomaly,
                    )
                )
            for lon in args.meridional_lons:
                slices.append(
                    extract_meridional_section(
                        inf.predicted_grid,
                        axes,
                        month,
                        label,
                        lon,
                        cora_grid=inf.cora_grid,
                        background_grid=inf.background_grid,
                        include_cora=include_cora,
                        include_anomaly=include_anomaly,
                    )
                )

            meta = {
                "split": split,
                "dataset_manifest_sha256": manifest_sha256,
                "architecture": lm.architecture if lm is not None else f"direct_{direct}",
                "checkpoint": str(lm.checkpoint) if lm is not None else "",
                "checkpoint_epoch": lm.epoch if lm is not None else "",
            }
            for slc in slices:
                rec = slice_summary_dict(slc)
                exported_files: list[str] = []
                if "nc" in args.export_data:
                    nc_path = export_slice_netcdf(slc, data_dir / f"{slc.slug}.nc", meta)
                    exported_files.append(f"slices/{nc_path.name}")
                if "npz" in args.export_data:
                    npz_path = export_slice_npz(slc, data_dir / f"{slc.slug}.npz", meta)
                    exported_files.append(f"slices/{npz_path.name}")
                saved_figs = plot_ocean_slice(slc, fig_dir, formats)
                fig_rels = [f"figures/{name}" for name in saved_figs]
                figures.extend(fig_rels)
                rec["data_files"] = exported_files
                rec["figures"] = fig_rels
                slice_records.append(rec)

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "mode": "slice",
        "device": str(device),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "manifest_sha256": manifest_sha256,
        "months": [month for month, _ in month_pairs],
        "depths_requested": args.depths,
        "zonal_lats_requested": args.zonal_lats,
        "meridional_lons_requested": args.meridional_lons,
        "export_data": args.export_data,
        "slices": slice_records,
        "notes": [
            "Land and invalid ocean cells are masked to NaN.",
            "Matched native grid coordinates are recorded alongside requested coordinates.",
            CORA_OVERLAP_NOTE,
        ],
    }
    (output_dir / "slice_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_report_readme(output_dir, "Ocean Temperature (CT) & Salinity (SA) Slice Report", manifest, figures)
    return output_dir


def _write_report_readme(
    output_dir: Path,
    title: str,
    payload: dict[str, Any],
    figures: list[str],
) -> None:
    """Write a concise markdown index inside a generated diagnostic or slice folder."""
    lines = [
        f"# {title}",
        "",
        f"- Created (UTC): `{payload['created_utc']}`",
        f"- Device: `{payload['device']}`",
        f"- Dataset manifest SHA-256: `{payload['manifest_sha256']}`",
        "",
        "## Notes",
        "",
        *[f"- {note}" for note in payload.get("notes", [])],
        "",
        "## Figures",
        "",
        *[f"- [{fig}]({fig})" for fig in figures],
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser with ``diagnose`` and ``slice`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="spatial_depth_diagnostics",
        description="Read-only spatial/depth diagnostics and 2D ocean CT/SA slice generator for SRTP v2 models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Subcommand 1: diagnose
    p_diag = sub.add_parser(
        "diagnose",
        help="Compute 5-500 m depth-wise error curves, spatial error maps, and representative Argo E profiles.",
    )
    p_diag.add_argument("--dataset-dir", type=Path, required=True, help="Path to training_ready_v2 directory")
    p_diag.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec as label=/path/to/best.pt (repeatable for multi-model comparison)",
    )
    p_diag.add_argument("--output-dir", type=Path, required=True, help="Fresh output directory (must not exist)")
    p_diag.add_argument(
        "--splits",
        nargs="+",
        choices=("validation", "test_locked", "forward_test"),
        default=["test_locked"],
        help="Dataset splits to diagnose (ignored when --months is set)",
    )
    p_diag.add_argument("--months", nargs="+", default=None, help="Optional explicit YYYYMM list (e.g. 202401 202407)")
    p_diag.add_argument(
        "--cohorts",
        nargs="+",
        choices=("all_E", "E_platforms_unseen_in_validation"),
        default=None,
        help="Optional Argo E cohorts to evaluate",
    )
    p_diag.add_argument("--bin-deg", type=float, default=2.0, help="Spatial binning width in degrees (default: 2.0)")
    p_diag.add_argument(
        "--spatial-bands",
        nargs="+",
        choices=tuple(name for name, _, _ in DEFAULT_DEPTH_BANDS),
        default=["full_5_500m", "thermocline_55_200m"],
        help="Depth bands for spatial error map figures (CSV always records all 4 bands)",
    )
    p_diag.add_argument("--num-profiles", type=int, default=6, help="Number of representative Argo E profiles to plot")
    p_diag.add_argument(
        "--reference-model",
        type=str,
        default=None,
        help="Model label used to stratify representative profile error quantiles (defaults to first --model)",
    )
    p_diag.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.75,
        help="Minimum valid depth fraction required for representative profile candidates",
    )
    p_diag.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p_diag.add_argument("--cpu-threads", type=int, default=4)
    p_diag.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf"])

    # Subcommand 2: slice
    p_slice = sub.add_parser(
        "slice",
        help="Extract and plot 2D horizontal depth slices and zonal/meridional vertical cross-sections of CT and SA.",
    )
    p_slice.add_argument("--dataset-dir", type=Path, required=True, help="Path to training_ready_v2 directory")
    p_slice.add_argument(
        "--model",
        action="append",
        default=[],
        help="Optional model spec label=/path/to/best.pt (repeatable)",
    )
    p_slice.add_argument(
        "--direct-source",
        nargs="+",
        choices=("cora", "background"),
        default=[],
        help="Optional direct ocean data source to slice without loading a neural checkpoint",
    )
    p_slice.add_argument("--output-dir", type=Path, required=True, help="Fresh output directory (must not exist)")
    p_slice.add_argument("--months", nargs="+", default=None, help="Target YYYYMM months (e.g. 202401 202407)")
    p_slice.add_argument(
        "--splits",
        nargs="+",
        choices=("validation", "test_locked", "forward_test"),
        default=None,
        help="Optional splits if --months is omitted",
    )
    p_slice.add_argument(
        "--depths", type=float, nargs="*", default=[], help="Horizontal slice depths in meters (5-500)"
    )
    p_slice.add_argument(
        "--zonal-lats",
        type=float,
        nargs="*",
        default=[],
        help="Zonal vertical cross-section latitudes in deg N (e.g. 0.0 5.0 -5.0)",
    )
    p_slice.add_argument(
        "--meridional-lons",
        type=float,
        nargs="*",
        default=[],
        help="Meridional vertical cross-section longitudes in deg E or negative deg W (e.g. 165 190 220 -110)",
    )
    p_slice.add_argument("--include-cora", action="store_true", help="Include CORA reference & (Source - CORA) panels")
    p_slice.add_argument(
        "--include-anomaly",
        action="store_true",
        help="Include (Source - Training Climatology) anomaly panels and arrays",
    )
    p_slice.add_argument(
        "--export-data",
        nargs="*",
        choices=("nc", "npz"),
        default=["nc", "npz"],
        help="Structured numerical formats to export under slices/ (default: nc npz)",
    )
    p_slice.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p_slice.add_argument("--cpu-threads", type=int, default=4)
    p_slice.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf"])

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for ``python -m spatial_depth_diagnostics``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = run_diagnose(args) if args.command == "diagnose" else run_slice(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(2, f"spatial_depth_diagnostics error: {error}\n")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
