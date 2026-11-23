"""Depth-wise, spatial-bin, and representative-profile diagnostic accumulators."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

VARIABLES: tuple[str, str] = ("CT", "SA")
DEFAULT_DEPTH_BANDS: tuple[tuple[str, float, float], ...] = (
    ("full_5_500m", 5.0, 500.0),
    ("surface_5_50m", 5.0, 50.0),
    ("thermocline_55_200m", 55.0, 200.0),
    ("subsurface_210_500m", 210.0, 500.0),
)


class DepthMetricsAccumulator:
    """Accumulate physical CT/SA error moments independently at each depth level."""

    def __init__(self, depths: np.ndarray) -> None:
        """Initialize float64 moment arrays for all native depth levels.

        Args:
            depths: 1D ascending array of depth centers in meters, shape ``(D,)``.
        """
        self.depths = np.asarray(depths, dtype=np.float64)
        self.num_depths = int(self.depths.size)
        # Dimensions: [variable (CT=0, SA=1), depth, moment]
        # Moments: 0=count, 1=obs_sum, 2=obs_sq_sum, 3=err_sum, 4=sse
        self.moments = np.zeros((2, self.num_depths, 5), dtype=np.float64)

    def add(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        valid: np.ndarray,
        cohort_mask: np.ndarray,
    ) -> None:
        """Accumulate valid Argo levels belonging to ``cohort_mask``.

        Args:
            observed: Observed physical values, shape ``(N, 2, D)``.
            predicted: Predicted physical values, shape ``(N, 2, D)``.
            valid: Boolean validity mask, shape ``(N, 2, D)``.
            cohort_mask: Boolean profile cohort selector, shape ``(N,)``.
        """
        if not np.any(cohort_mask):
            return
        obs_c = observed[cohort_mask].astype(np.float64)
        pred_c = predicted[cohort_mask].astype(np.float64)
        val_c = valid[cohort_mask].astype(bool) & np.isfinite(obs_c) & np.isfinite(pred_c)
        for var_idx in range(2):
            mask = val_c[:, var_idx, :]
            obs_v = np.where(mask, obs_c[:, var_idx, :], 0.0)
            err_v = np.where(mask, pred_c[:, var_idx, :] - obs_c[:, var_idx, :], 0.0)
            self.moments[var_idx, :, 0] += mask.sum(axis=0)
            self.moments[var_idx, :, 1] += obs_v.sum(axis=0)
            self.moments[var_idx, :, 2] += np.square(obs_v).sum(axis=0)
            self.moments[var_idx, :, 3] += err_v.sum(axis=0)
            self.moments[var_idx, :, 4] += np.square(err_v).sum(axis=0)

    def to_rows(self, model: str, split: str, cohort: str) -> list[dict[str, Any]]:
        """Return per-depth metric records for CSV and JSON serialization."""
        rows: list[dict[str, Any]] = []
        for var_idx, variable in enumerate(VARIABLES):
            for depth_idx, depth_m in enumerate(self.depths):
                count, obs_sum, obs_sq_sum, err_sum, sse = self.moments[var_idx, depth_idx]
                valid_points = int(count)
                if valid_points > 0:
                    rmse: float | None = math.sqrt(sse / count)
                    bias: float | None = float(err_sum / count)
                    sst = obs_sq_sum - (obs_sum * obs_sum) / count
                    r2: float | None = float(1.0 - sse / sst) if sst > 0.0 else None
                else:
                    rmse, bias, r2 = None, None, None
                rows.append(
                    {
                        "model": model,
                        "split": split,
                        "cohort": cohort,
                        "variable": variable,
                        "depth_m": float(depth_m),
                        "valid_points": valid_points,
                        "rmse": rmse,
                        "bias": bias,
                        "r2": r2,
                    }
                )
        return rows


class SpatialBinAccumulator:
    """Accumulate Argo E RMSE and bias on regular latitude/longitude bins per depth band."""

    def __init__(
        self,
        lat_axis: np.ndarray,
        lon_axis: np.ndarray,
        depths: np.ndarray,
        bin_deg: float = 2.0,
        depth_bands: tuple[tuple[str, float, float], ...] = DEFAULT_DEPTH_BANDS,
    ) -> None:
        """Create spatial bin edges spanning the native grid domain.

        Args:
            lat_axis: Native latitude centers in degrees north.
            lon_axis: Native longitude centers in degrees east.
            depths: Native depth centers in meters.
            bin_deg: Bin width in degrees (must be positive).
            depth_bands: Sequence of ``(band_name, min_depth_m, max_depth_m)``.
        """
        if bin_deg <= 0.0:
            raise ValueError(f"bin_deg must be positive, got {bin_deg}")
        self.bin_deg = float(bin_deg)
        lat_min = math.floor(float(lat_axis.min()) / bin_deg) * bin_deg
        lat_max = math.ceil(float(lat_axis.max()) / bin_deg) * bin_deg
        lon_min = math.floor(float(lon_axis.min()) / bin_deg) * bin_deg
        lon_max = math.ceil(float(lon_axis.max()) / bin_deg) * bin_deg
        n_lat = max(1, int(round((lat_max - lat_min) / bin_deg)))
        n_lon = max(1, int(round((lon_max - lon_min) / bin_deg)))
        self.lat_edges = np.linspace(lat_min, lat_max, n_lat + 1, dtype=np.float64)
        self.lon_edges = np.linspace(lon_min, lon_max, n_lon + 1, dtype=np.float64)
        self.lat_centers = 0.5 * (self.lat_edges[:-1] + self.lat_edges[1:])
        self.lon_centers = 0.5 * (self.lon_edges[:-1] + self.lon_edges[1:])
        self.depths = np.asarray(depths, dtype=np.float64)
        self.depth_bands = depth_bands
        self.band_masks: dict[str, np.ndarray] = {
            name: (self.depths >= d_min) & (self.depths <= d_max) for name, d_min, d_max in depth_bands
        }
        # Per band: moments [var(2), n_lat, n_lon, 3(count, err_sum, sse)] and profile_counts [n_lat, n_lon]
        self.moments: dict[str, np.ndarray] = {
            name: np.zeros((2, n_lat, n_lon, 3), dtype=np.float64) for name, _, _ in depth_bands
        }
        self.profile_counts: dict[str, np.ndarray] = {
            name: np.zeros((n_lat, n_lon), dtype=np.int64) for name, _, _ in depth_bands
        }

    def add(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        valid: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        cohort_mask: np.ndarray,
    ) -> None:
        """Bin valid Argo profile errors into geographic cells for each depth band."""
        if not np.any(cohort_mask):
            return
        obs_c = observed[cohort_mask].astype(np.float64)
        pred_c = predicted[cohort_mask].astype(np.float64)
        val_c = valid[cohort_mask].astype(bool) & np.isfinite(obs_c) & np.isfinite(pred_c)
        lat_c = np.asarray(lat[cohort_mask], dtype=np.float64)
        lon_c = np.asarray(lon[cohort_mask], dtype=np.float64)
        iy = np.clip(np.searchsorted(self.lat_edges, lat_c, side="right") - 1, 0, self.lat_centers.size - 1)
        ix = np.clip(np.searchsorted(self.lon_edges, lon_c, side="right") - 1, 0, self.lon_centers.size - 1)

        for band_name, d_mask in self.band_masks.items():
            if not np.any(d_mask):
                continue
            band_valid = val_c[:, :, d_mask]
            has_profile = band_valid.any(axis=(1, 2))
            if np.any(has_profile):
                np.add.at(self.profile_counts[band_name], (iy[has_profile], ix[has_profile]), 1)
            band_err = pred_c[:, :, d_mask] - obs_c[:, :, d_mask]
            for var_idx in range(2):
                v_mask = band_valid[:, var_idx, :]
                counts = v_mask.sum(axis=1).astype(np.float64)
                err_sum = np.where(v_mask, band_err[:, var_idx, :], 0.0).sum(axis=1)
                sse = np.where(v_mask, np.square(band_err[:, var_idx, :]), 0.0).sum(axis=1)
                np.add.at(self.moments[band_name][var_idx, :, :, 0], (iy, ix), counts)
                np.add.at(self.moments[band_name][var_idx, :, :, 1], (iy, ix), err_sum)
                np.add.at(self.moments[band_name][var_idx, :, :, 2], (iy, ix), sse)

    def grid_maps(self, band_name: str) -> dict[str, np.ndarray]:
        """Return 2D ``(n_lat, n_lon)`` arrays of RMSE, Bias, and counts (empty cells as NaN)."""
        m = self.moments[band_name]
        result: dict[str, np.ndarray] = {"profiles": self.profile_counts[band_name].copy()}
        for var_idx, name in enumerate(("ct", "sa")):
            count = m[var_idx, :, :, 0]
            err_sum = m[var_idx, :, :, 1]
            sse = m[var_idx, :, :, 2]
            rmse = np.full_like(count, np.nan, dtype=np.float64)
            bias = np.full_like(count, np.nan, dtype=np.float64)
            valid = count > 0
            rmse[valid] = np.sqrt(sse[valid] / count[valid])
            bias[valid] = err_sum[valid] / count[valid]
            result[f"{name}_valid_points"] = count.astype(np.int64)
            result[f"{name}_rmse"] = rmse
            result[f"{name}_bias"] = bias
        return result

    def to_rows(self, model: str, split: str, cohort: str) -> list[dict[str, Any]]:
        """Export non-empty spatial bins for CSV persistence."""
        rows: list[dict[str, Any]] = []
        for band_name, _, _ in self.depth_bands:
            maps = self.grid_maps(band_name)
            for iy, lat_val in enumerate(self.lat_centers):
                for ix, lon_val in enumerate(self.lon_centers):
                    ct_n = int(maps["ct_valid_points"][iy, ix])
                    sa_n = int(maps["sa_valid_points"][iy, ix])
                    if ct_n == 0 and sa_n == 0:
                        continue
                    rows.append(
                        {
                            "model": model,
                            "split": split,
                            "cohort": cohort,
                            "depth_band": band_name,
                            "lat_center": float(lat_val),
                            "lon_center": float(lon_val),
                            "profiles": int(maps["profiles"][iy, ix]),
                            "ct_valid_points": ct_n,
                            "ct_rmse": None if ct_n == 0 else float(maps["ct_rmse"][iy, ix]),
                            "ct_bias": None if ct_n == 0 else float(maps["ct_bias"][iy, ix]),
                            "sa_valid_points": sa_n,
                            "sa_rmse": None if sa_n == 0 else float(maps["sa_rmse"][iy, ix]),
                            "sa_bias": None if sa_n == 0 else float(maps["sa_bias"][iy, ix]),
                        }
                    )
        return rows


@dataclass
class ProfileCandidate:
    """One held-out Argo E profile paired with climatology and multi-model reconstructions."""

    key: tuple[str, int]
    month: str
    profile_index: int
    platform: str
    latitude: float
    longitude: float
    time_offset_days: float
    observed_ct: np.ndarray
    observed_sa: np.ndarray
    background_ct: np.ndarray
    background_sa: np.ndarray
    valid_ct: np.ndarray
    valid_sa: np.ndarray
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def model_rmse(self, model_label: str) -> tuple[float, float, float]:
        """Return ``(ct_rmse, sa_rmse, combined_score)`` for one model on this profile."""
        pred_ct, pred_sa = self.predictions[model_label]
        ct_err = pred_ct[self.valid_ct] - self.observed_ct[self.valid_ct]
        sa_err = pred_sa[self.valid_sa] - self.observed_sa[self.valid_sa]
        ct_rmse = float(np.sqrt(np.mean(np.square(ct_err)))) if ct_err.size else math.nan
        sa_rmse = float(np.sqrt(np.mean(np.square(sa_err)))) if sa_err.size else math.nan
        # Combine CT and SA using approximate physical scale ratio (1.0 C vs 0.15 g/kg)
        combined = float(ct_rmse + sa_rmse / 0.15)
        return ct_rmse, sa_rmse, combined


class RepresentativeProfileCollector:
    """Collect candidate Argo E profiles across models and choose stratified examples."""

    def __init__(self, min_valid_fraction: float = 0.75) -> None:
        """Initialize the collector.

        Args:
            min_valid_fraction: Minimum fraction of depth levels required for both CT and SA.
        """
        self.min_valid_fraction = float(min_valid_fraction)
        self._candidates: dict[tuple[str, int], ProfileCandidate] = {}

    def add_month(
        self,
        model_label: str,
        month: str,
        observed: np.ndarray,
        predicted: np.ndarray,
        background: np.ndarray,
        valid: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        time_offset: np.ndarray,
        platform: np.ndarray,
        cohort_mask: np.ndarray,
    ) -> None:
        """Record eligible profiles from one month and model."""
        num_depths = observed.shape[2]
        min_levels = max(2, int(math.ceil(num_depths * self.min_valid_fraction)))
        indices = np.flatnonzero(cohort_mask)
        for idx_np in indices:
            idx = int(idx_np)
            v_ct = valid[idx, 0].astype(bool) & np.isfinite(observed[idx, 0]) & np.isfinite(predicted[idx, 0])
            v_sa = valid[idx, 1].astype(bool) & np.isfinite(observed[idx, 1]) & np.isfinite(predicted[idx, 1])
            if int(v_ct.sum()) < min_levels or int(v_sa.sum()) < min_levels:
                continue
            key = (month, idx)
            if key not in self._candidates:
                self._candidates[key] = ProfileCandidate(
                    key=key,
                    month=month,
                    profile_index=idx,
                    platform=str(platform[idx]),
                    latitude=float(lat[idx]),
                    longitude=float(lon[idx]),
                    time_offset_days=float(time_offset[idx]),
                    observed_ct=observed[idx, 0].astype(np.float32).copy(),
                    observed_sa=observed[idx, 1].astype(np.float32).copy(),
                    background_ct=background[idx, 0].astype(np.float32).copy(),
                    background_sa=background[idx, 1].astype(np.float32).copy(),
                    valid_ct=v_ct.copy(),
                    valid_sa=v_sa.copy(),
                )
            self._candidates[key].predictions[model_label] = (
                predicted[idx, 0].astype(np.float32).copy(),
                predicted[idx, 1].astype(np.float32).copy(),
            )

    def select(
        self,
        reference_model: str,
        required_models: list[str],
        count: int = 6,
    ) -> list[dict[str, Any]]:
        """Select stratified representative profiles from best to challenging error quantiles."""
        eligible = [
            cand for cand in self._candidates.values() if all(model in cand.predictions for model in required_models)
        ]
        if not eligible or count <= 0:
            return []
        scored: list[tuple[float, float, float, ProfileCandidate]] = []
        for cand in eligible:
            ct_rmse, sa_rmse, combined = cand.model_rmse(reference_model)
            if math.isfinite(combined):
                scored.append((combined, ct_rmse, sa_rmse, cand))
        if not scored:
            return []
        scored.sort(key=lambda item: (item[0], item[3].month, item[3].profile_index))
        n = len(scored)
        if n <= count:
            chosen_indices = list(range(n))
        else:
            quantiles = np.linspace(0.08, 0.92, count)
            chosen_indices = []
            used: set[int] = set()
            for q in quantiles:
                idx = int(round(q * (n - 1)))
                while idx in used and idx + 1 < n:
                    idx += 1
                while idx in used and idx > 0:
                    idx -= 1
                if idx not in used:
                    used.add(idx)
                    chosen_indices.append(idx)

        selected: list[dict[str, Any]] = []
        for rank_idx, idx in enumerate(chosen_indices):
            combined, ct_rmse, sa_rmse, cand = scored[idx]
            percentile = 100.0 * idx / max(1, n - 1)
            selected.append(
                {
                    "rank": rank_idx + 1,
                    "percentile": round(percentile, 1),
                    "reference_model": reference_model,
                    "reference_ct_rmse": ct_rmse,
                    "reference_sa_rmse": sa_rmse,
                    "reference_combined_score": combined,
                    "month": cand.month,
                    "profile_index": cand.profile_index,
                    "platform": cand.platform,
                    "latitude": cand.latitude,
                    "longitude": cand.longitude,
                    "time_offset_days": cand.time_offset_days,
                    "observed_ct": np.where(cand.valid_ct, cand.observed_ct, np.nan).tolist(),
                    "observed_sa": np.where(cand.valid_sa, cand.observed_sa, np.nan).tolist(),
                    "background_ct": np.where(cand.valid_ct, cand.background_ct, np.nan).tolist(),
                    "background_sa": np.where(cand.valid_sa, cand.background_sa, np.nan).tolist(),
                    "predictions": {
                        model: {
                            "CT": np.where(cand.valid_ct, cand.predictions[model][0], np.nan).tolist(),
                            "SA": np.where(cand.valid_sa, cand.predictions[model][1], np.nan).tolist(),
                            "ct_rmse": cand.model_rmse(model)[0],
                            "sa_rmse": cand.model_rmse(model)[1],
                        }
                        for model in required_models
                    },
                }
            )
        return selected
