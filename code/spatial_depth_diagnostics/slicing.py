"""2D ocean temperature (CT) and salinity (SA) slice extraction and data export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from netCDF4 import Dataset

from spatial_depth_diagnostics.loader import GridAxes

SliceKind = Literal["horizontal", "zonal", "meridional"]


@dataclass(frozen=True)
class SliceCoordinate:
    """Exact mapping between a user-requested slice coordinate and the native grid axis."""

    axis_name: str
    unit: str
    requested: float
    matched: float
    index: int


@dataclass(frozen=True)
class OceanSlice:
    """One 2D slice of CT and SA fields with coordinates and optional reference/anomaly arrays."""

    kind: SliceKind
    month: str
    source_label: str
    fixed_coord: SliceCoordinate
    x_name: str
    x_unit: str
    x_values: np.ndarray
    y_name: str
    y_unit: str
    y_values: np.ndarray
    ct: np.ndarray
    sa: np.ndarray
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        """Return a filesystem-safe identifier for this slice."""
        coord_tag = f"{self.fixed_coord.matched:g}".replace("-", "m").replace(".", "p")
        safe_source = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in self.source_label)
        return f"{safe_source}_{self.month}_{self.kind}_{self.fixed_coord.axis_name}_{coord_tag}"


def normalize_longitude(requested_lon: float, lon_axis: np.ndarray) -> float:
    """Normalize negative west longitudes into ``[0, 360)`` when the grid uses 0-360E."""
    lon = float(requested_lon)
    if float(lon_axis.min()) >= 0.0 and lon < 0.0:
        lon = lon % 360.0
    return lon


def match_axis_index(
    axis: np.ndarray,
    requested: float,
    axis_name: str,
    unit: str,
) -> SliceCoordinate:
    """Find the nearest native grid index and validate that the request lies within domain bounds."""
    val = normalize_longitude(requested, axis) if axis_name == "longitude" else float(requested)
    low, high = float(axis[0]), float(axis[-1])
    # Allow a small tolerance equal to half the edge grid spacing
    tol = 0.5 * float(max(abs(axis[1] - axis[0]), abs(axis[-1] - axis[-2])))
    if val < low - tol or val > high + tol:
        raise ValueError(
            f"Requested {axis_name}={requested} ({val} {unit}) is outside native grid range [{low:.3f}, {high:.3f}]"
        )
    idx = int(np.argmin(np.abs(axis - val)))
    return SliceCoordinate(
        axis_name=axis_name,
        unit=unit,
        requested=float(requested),
        matched=float(axis[idx]),
        index=idx,
    )


def _slice_optional_extras(
    slicer: tuple[int | slice, ...],
    ct_main: np.ndarray,
    sa_main: np.ndarray,
    cora_grid: np.ndarray | None,
    background_grid: np.ndarray | None,
    include_cora: bool,
    include_anomaly: bool,
) -> dict[str, np.ndarray]:
    """Build optional CORA reference, difference, and climatology anomaly 2D arrays."""
    extras: dict[str, np.ndarray] = {}
    if include_cora and cora_grid is not None:
        ct_cora = cora_grid[(0, *slicer)].astype(np.float32)
        sa_cora = cora_grid[(1, *slicer)].astype(np.float32)
        extras["CT_cora"] = ct_cora
        extras["SA_cora"] = sa_cora
        extras["CT_diff_cora"] = (ct_main - ct_cora).astype(np.float32)
        extras["SA_diff_cora"] = (sa_main - sa_cora).astype(np.float32)
    if include_anomaly and background_grid is not None:
        ct_bg = background_grid[(0, *slicer)].astype(np.float32)
        sa_bg = background_grid[(1, *slicer)].astype(np.float32)
        extras["CT_background"] = ct_bg
        extras["SA_background"] = sa_bg
        extras["CT_anomaly"] = (ct_main - ct_bg).astype(np.float32)
        extras["SA_anomaly"] = (sa_main - sa_bg).astype(np.float32)
    return extras


def extract_horizontal_slice(
    grid: np.ndarray,
    axes: GridAxes,
    month: str,
    source_label: str,
    requested_depth: float,
    cora_grid: np.ndarray | None = None,
    background_grid: np.ndarray | None = None,
    include_cora: bool = False,
    include_anomaly: bool = False,
) -> OceanSlice:
    """Extract a horizontal ``(latitude, longitude)`` CT/SA slice at a target depth."""
    coord = match_axis_index(axes.depth, requested_depth, "depth", "m")
    slicer: tuple[int | slice, ...] = (coord.index, slice(None), slice(None))
    ct = grid[(0, *slicer)].astype(np.float32)
    sa = grid[(1, *slicer)].astype(np.float32)
    extras = _slice_optional_extras(slicer, ct, sa, cora_grid, background_grid, include_cora, include_anomaly)
    return OceanSlice(
        kind="horizontal",
        month=month,
        source_label=source_label,
        fixed_coord=coord,
        x_name="longitude",
        x_unit="degrees_east",
        x_values=axes.longitude.copy(),
        y_name="latitude",
        y_unit="degrees_north",
        y_values=axes.latitude.copy(),
        ct=ct,
        sa=sa,
        extras=extras,
    )


def extract_zonal_section(
    grid: np.ndarray,
    axes: GridAxes,
    month: str,
    source_label: str,
    requested_lat: float,
    cora_grid: np.ndarray | None = None,
    background_grid: np.ndarray | None = None,
    include_cora: bool = False,
    include_anomaly: bool = False,
) -> OceanSlice:
    """Extract a zonal vertical cross-section ``(depth, longitude)`` along a target latitude."""
    coord = match_axis_index(axes.latitude, requested_lat, "latitude", "degrees_north")
    slicer: tuple[int | slice, ...] = (slice(None), coord.index, slice(None))
    ct = grid[(0, *slicer)].astype(np.float32)
    sa = grid[(1, *slicer)].astype(np.float32)
    extras = _slice_optional_extras(slicer, ct, sa, cora_grid, background_grid, include_cora, include_anomaly)
    return OceanSlice(
        kind="zonal",
        month=month,
        source_label=source_label,
        fixed_coord=coord,
        x_name="longitude",
        x_unit="degrees_east",
        x_values=axes.longitude.copy(),
        y_name="depth",
        y_unit="m",
        y_values=axes.depth.copy(),
        ct=ct,
        sa=sa,
        extras=extras,
    )


def extract_meridional_section(
    grid: np.ndarray,
    axes: GridAxes,
    month: str,
    source_label: str,
    requested_lon: float,
    cora_grid: np.ndarray | None = None,
    background_grid: np.ndarray | None = None,
    include_cora: bool = False,
    include_anomaly: bool = False,
) -> OceanSlice:
    """Extract a meridional vertical cross-section ``(depth, latitude)`` along a target longitude."""
    coord = match_axis_index(axes.longitude, requested_lon, "longitude", "degrees_east")
    slicer: tuple[int | slice, ...] = (slice(None), slice(None), coord.index)
    ct = grid[(0, *slicer)].astype(np.float32)
    sa = grid[(1, *slicer)].astype(np.float32)
    extras = _slice_optional_extras(slicer, ct, sa, cora_grid, background_grid, include_cora, include_anomaly)
    return OceanSlice(
        kind="meridional",
        month=month,
        source_label=source_label,
        fixed_coord=coord,
        x_name="latitude",
        x_unit="degrees_north",
        x_values=axes.latitude.copy(),
        y_name="depth",
        y_unit="m",
        y_values=axes.depth.copy(),
        ct=ct,
        sa=sa,
        extras=extras,
    )


_VARIABLE_META: dict[str, tuple[str, str]] = {
    "CT": ("degC", "Conservative Temperature"),
    "SA": ("g/kg", "Absolute Salinity"),
    "CT_cora": ("degC", "CORA Conservative Temperature Reference"),
    "SA_cora": ("g/kg", "CORA Absolute Salinity Reference"),
    "CT_diff_cora": ("degC", "Conservative Temperature Difference (Source minus CORA)"),
    "SA_diff_cora": ("g/kg", "Absolute Salinity Difference (Source minus CORA)"),
    "CT_background": ("degC", "Training Monthly Climatology Conservative Temperature"),
    "SA_background": ("g/kg", "Training Monthly Climatology Absolute Salinity"),
    "CT_anomaly": ("degC", "Conservative Temperature Anomaly relative to Training Climatology"),
    "SA_anomaly": ("g/kg", "Absolute Salinity Anomaly relative to Training Climatology"),
}


def export_slice_netcdf(ocean_slice: OceanSlice, path: Path, metadata: dict[str, Any]) -> Path:
    """Write an OceanSlice to a self-describing NetCDF4 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, mode="w", format="NETCDF4") as ds:
        ds.createDimension(ocean_slice.y_name, int(ocean_slice.y_values.size))
        ds.createDimension(ocean_slice.x_name, int(ocean_slice.x_values.size))

        y_var = ds.createVariable(ocean_slice.y_name, "f8", (ocean_slice.y_name,))
        y_var[:] = ocean_slice.y_values
        y_var.units = ocean_slice.y_unit
        if ocean_slice.y_name == "depth":
            y_var.positive = "down"

        x_var = ds.createVariable(ocean_slice.x_name, "f8", (ocean_slice.x_name,))
        x_var[:] = ocean_slice.x_values
        x_var.units = ocean_slice.x_unit

        arrays: dict[str, np.ndarray] = {"CT": ocean_slice.ct, "SA": ocean_slice.sa, **ocean_slice.extras}
        for name, values in arrays.items():
            var = ds.createVariable(
                name,
                "f4",
                (ocean_slice.y_name, ocean_slice.x_name),
                zlib=True,
                complevel=4,
                fill_value=np.float32(np.nan),
            )
            var[:, :] = values.astype(np.float32)
            units, long_name = _VARIABLE_META.get(name, ("", name))
            var.units = units
            var.long_name = long_name

        ds.slice_kind = ocean_slice.kind
        ds.month = ocean_slice.month
        ds.source_label = ocean_slice.source_label
        ds.fixed_axis = ocean_slice.fixed_coord.axis_name
        ds.fixed_axis_unit = ocean_slice.fixed_coord.unit
        ds.fixed_requested = float(ocean_slice.fixed_coord.requested)
        ds.fixed_matched = float(ocean_slice.fixed_coord.matched)
        ds.fixed_index = int(ocean_slice.fixed_coord.index)
        for key, value in metadata.items():
            if value is not None:
                setattr(ds, str(key), str(value))
    return path


def export_slice_npz(ocean_slice: OceanSlice, path: Path, metadata: dict[str, Any]) -> Path:
    """Write an OceanSlice to a compressed NumPy ``.npz`` archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        ocean_slice.y_name: ocean_slice.y_values.astype(np.float64),
        ocean_slice.x_name: ocean_slice.x_values.astype(np.float64),
        "CT": ocean_slice.ct.astype(np.float32),
        "SA": ocean_slice.sa.astype(np.float32),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "slice_kind": ocean_slice.kind,
                    "month": ocean_slice.month,
                    "source_label": ocean_slice.source_label,
                    "fixed_axis": ocean_slice.fixed_coord.axis_name,
                    "fixed_axis_unit": ocean_slice.fixed_coord.unit,
                    "fixed_requested": ocean_slice.fixed_coord.requested,
                    "fixed_matched": ocean_slice.fixed_coord.matched,
                    "fixed_index": ocean_slice.fixed_coord.index,
                    **metadata,
                },
                ensure_ascii=False,
            )
        ),
    }
    for name, values in ocean_slice.extras.items():
        payload[name] = values.astype(np.float32)
    np.savez_compressed(path, **payload)
    return path


def slice_summary_dict(ocean_slice: OceanSlice) -> dict[str, Any]:
    """Return summary min/max/mean statistics of an OceanSlice for manifest logging."""

    def stats(arr: np.ndarray) -> dict[str, float | int | None]:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return {"valid_cells": 0, "min": None, "max": None, "mean": None}
        return {
            "valid_cells": int(finite.size),
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
        }

    return {
        "slug": ocean_slice.slug,
        "kind": ocean_slice.kind,
        "month": ocean_slice.month,
        "source_label": ocean_slice.source_label,
        "fixed_axis": ocean_slice.fixed_coord.axis_name,
        "fixed_unit": ocean_slice.fixed_coord.unit,
        "fixed_requested": ocean_slice.fixed_coord.requested,
        "fixed_matched": ocean_slice.fixed_coord.matched,
        "fixed_index": ocean_slice.fixed_coord.index,
        "shape": [int(ocean_slice.y_values.size), int(ocean_slice.x_values.size)],
        "CT": stats(ocean_slice.ct),
        "SA": stats(ocean_slice.sa),
        "extras": {name: stats(arr) for name, arr in ocean_slice.extras.items()},
    }
