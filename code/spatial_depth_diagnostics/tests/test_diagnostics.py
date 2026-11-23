"""Synthetic unit tests for depth/spatial diagnostics, ocean CT/SA slicing, and plotting."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from spatial_depth_diagnostics.loader import GridAxes, infer_architecture, parse_model_spec
from spatial_depth_diagnostics.metrics import (
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
    export_slice_netcdf,
    export_slice_npz,
    extract_horizontal_slice,
    extract_meridional_section,
    extract_zonal_section,
    match_axis_index,
    normalize_longitude,
    slice_summary_dict,
)


def _synthetic_axes() -> GridAxes:
    """Build non-uniform Tropical Pacific-like coordinates matching v2 structure."""
    depth = np.concatenate((np.arange(5.0, 105.0, 5.0), np.arange(110.0, 510.0, 10.0)))
    # Non-uniform Mercator-like latitude spacing
    s = np.linspace(-1.0, 1.0, 41)
    latitude = 19.6 * np.sinh(0.8 * s) / np.sinh(0.8)
    longitude = np.linspace(120.0, 280.0, 81)
    return GridAxes(depth=depth, latitude=latitude, longitude=longitude)


def _synthetic_ocean_fields(axes: GridAxes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate realistic synthetic Tropical Pacific CT/SA fields with land masks."""
    d = axes.depth[:, None, None]
    lat = axes.latitude[None, :, None]
    lon = axes.longitude[None, None, :]

    # Realistic east-west tilting thermocline: deeper warm pool in west, shallower cold tongue in east
    thermocline_depth = 160.0 - 0.55 * (lon - 120.0)
    ct_bg = 8.0 + 21.0 / (1.0 + np.exp((d - thermocline_depth) / 45.0)) - 0.015 * (lat**2)
    sa_bg = 34.4 + 0.9 * np.exp(-(((d - 120.0) / 110.0) ** 2)) + 0.004 * (lon - 200.0) * np.exp(-((lat / 10.0) ** 2))

    # Realistic anomaly and slight model error
    ct_anom = 1.4 * np.sin((lon - 120.0) * math.pi / 160.0) * np.exp(-(((d - 80.0) / 120.0) ** 2))
    sa_anom = -0.25 * np.cos((lon - 120.0) * math.pi / 160.0) * np.exp(-(((d - 60.0) / 100.0) ** 2))

    cora = np.stack((ct_bg + ct_anom, sa_bg + sa_anom), axis=0).astype(np.float32)
    pred = np.stack((ct_bg + 0.9 * ct_anom + 0.15, sa_bg + 0.85 * sa_anom - 0.03), axis=0).astype(np.float32)
    bg = np.stack((ct_bg, sa_bg), axis=0).astype(np.float32)

    # Add synthetic Maritime Continent & Americas land corners as NaN
    land = ((axes.latitude[:, None] < -10.0) & (axes.longitude[None, :] < 135.0)) | (
        (axes.latitude[:, None] > 12.0) & (axes.longitude[None, :] > 265.0)
    )
    for arr in (cora, pred, bg):
        arr[:, :, land] = np.nan
    return pred, cora, bg


class LoaderUtilsTest(unittest.TestCase):
    """Verify pure helper contracts in loader.py."""

    def test_parse_model_spec(self) -> None:
        spec = parse_model_spec("swin=/workspace/srtp-ocean/runs/test/best.pt")
        self.assertEqual(spec.label, "swin")
        self.assertEqual(spec.checkpoint, Path("/workspace/srtp-ocean/runs/test/best.pt"))
        inferred = parse_model_spec("/tmp/residual_cnn/best.pt")
        self.assertEqual(inferred.label, "residual_cnn")
        with self.assertRaises(ValueError):
            parse_model_spec("   ")

    def test_infer_architecture(self) -> None:
        self.assertEqual(
            infer_architecture({"input_channels": 276, "dim": 48, "blocks": 4, "heads": 4, "window": 8}),
            "swin",
        )
        self.assertEqual(
            infer_architecture({"schema": "ocean_baseline_checkpoint_v1", "architecture": "residual_cnn"}),
            "residual_cnn",
        )
        with self.assertRaises(ValueError):
            infer_architecture({"schema": "unknown"})


class MetricsAndPlotsTest(unittest.TestCase):
    """Verify depth metrics, spatial binning, profile selection, and figure rendering."""

    def test_depth_metrics_exact_values(self) -> None:
        depths = np.array([5.0, 10.0, 20.0])
        acc = DepthMetricsAccumulator(depths)
        obs = np.array(
            [
                [[20.0, 18.0, 15.0], [34.0, 34.5, 35.0]],
                [[24.0, 22.0, 19.0], [35.0, 35.5, 36.0]],
            ],
            dtype=np.float64,
        )
        pred = obs.copy()
        pred[:, 0, 0] += 1.0  # CT depth 5m: bias=+1.0, rmse=1.0
        pred[:, 1, 2] -= 0.2  # SA depth 20m: bias=-0.2, rmse=0.2
        valid = np.ones_like(obs, dtype=bool)
        valid[:, 0, 1] = False  # No valid CT at 10m
        acc.add(obs, pred, valid, np.array([True, True]))

        rows = acc.to_rows("swin", "test_locked", "all_E")
        by_key = {(r["variable"], r["depth_m"]): r for r in rows}
        self.assertAlmostEqual(by_key[("CT", 5.0)]["rmse"], 1.0, places=6)
        self.assertAlmostEqual(by_key[("CT", 5.0)]["bias"], 1.0, places=6)
        self.assertAlmostEqual(by_key[("CT", 5.0)]["r2"], 0.75, places=6)
        self.assertEqual(by_key[("CT", 10.0)]["valid_points"], 0)
        self.assertIsNone(by_key[("CT", 10.0)]["rmse"])
        self.assertAlmostEqual(by_key[("SA", 20.0)]["rmse"], 0.2, places=6)
        self.assertAlmostEqual(by_key[("SA", 20.0)]["bias"], -0.2, places=6)

    def test_end_to_end_diagnostics_and_plots(self) -> None:
        axes = _synthetic_axes()
        rng = np.random.default_rng(20261120)
        n_prof = 80
        n_depth = axes.depth.size
        lats = rng.uniform(-18.0, 18.0, size=n_prof)
        lons = rng.uniform(125.0, 275.0, size=n_prof)
        times = rng.uniform(-15.0, 15.0, size=n_prof)
        platforms = np.array([f"590{idx:04d}" for idx in range(n_prof)], dtype=str)

        base_ct = 10.0 + 18.0 * np.exp(-axes.depth / 150.0)[None, :]
        base_sa = 34.5 + 0.6 * np.exp(-(((axes.depth - 100.0) / 80.0) ** 2))[None, :]
        obs = np.stack(
            (
                base_ct + rng.normal(0.0, 1.5, size=(n_prof, n_depth)),
                base_sa + rng.normal(0.0, 0.2, size=(n_prof, n_depth)),
            ),
            axis=1,
        ).astype(np.float32)
        bg = np.stack(
            (np.broadcast_to(base_ct, (n_prof, n_depth)), np.broadcast_to(base_sa, (n_prof, n_depth))), axis=1
        )
        pred_swin = obs + np.stack(
            (rng.normal(0.0, 0.6, size=(n_prof, n_depth)), rng.normal(0.0, 0.08, size=(n_prof, n_depth))),
            axis=1,
        ).astype(np.float32)
        pred_cnn = obs + np.stack(
            (rng.normal(0.05, 0.75, size=(n_prof, n_depth)), rng.normal(-0.02, 0.10, size=(n_prof, n_depth))),
            axis=1,
        ).astype(np.float32)
        valid = np.ones_like(obs, dtype=bool)
        cohort = np.ones(n_prof, dtype=bool)

        depth_rows = []
        collector = RepresentativeProfileCollector(min_valid_fraction=0.75)
        spatial_swin = SpatialBinAccumulator(axes.latitude, axes.longitude, axes.depth, bin_deg=4.0)

        for label, pred in (("swin", pred_swin), ("cnn", pred_cnn)):
            d_acc = DepthMetricsAccumulator(axes.depth)
            d_acc.add(obs, pred, valid, cohort)
            depth_rows.extend(d_acc.to_rows(label, "test_locked", "all_E"))
            collector.add_month(label, "202401", obs, pred, bg, valid, lats, lons, times, platforms, cohort)
            if label == "swin":
                spatial_swin.add(obs, pred, valid, lats, lons, cohort)

        selected = collector.select("swin", ["swin", "cnn"], count=4)
        self.assertEqual(len(selected), 4)
        self.assertTrue(selected[0]["reference_combined_score"] <= selected[-1]["reference_combined_score"])
        self.assertGreater(len(spatial_swin.to_rows("swin", "test_locked", "all_E")), 0)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            f_depth = plot_depth_error_curves(depth_rows, "test_locked", "all_E", out_dir, ("png", "pdf"))
            f_spatial = plot_spatial_error_maps(
                spatial_swin, "swin", "test_locked", "all_E", "full_5_500m", out_dir, ("png",)
            )
            f_prof = plot_representative_profiles(selected, axes.depth, "test_locked", "all_E", out_dir, ("png",))
            for rel in f_depth + f_spatial + f_prof:
                path = out_dir / rel
                self.assertTrue(path.is_file(), f"Missing plot: {path}")
                self.assertGreater(path.stat().st_size, 1000)


class OceanSlicingTest(unittest.TestCase):
    """Verify horizontal, zonal, and meridional CT/SA slice extraction, NetCDF/NPZ export, and plotting."""

    def test_longitude_and_bounds(self) -> None:
        axes = _synthetic_axes()
        self.assertAlmostEqual(normalize_longitude(-140.0, axes.longitude), 220.0)
        coord = match_axis_index(axes.longitude, -140.0, "longitude", "degrees_east")
        self.assertAlmostEqual(coord.matched, 220.0)
        with self.assertRaises(ValueError):
            match_axis_index(axes.depth, 900.0, "depth", "m")

    def test_slice_extraction_export_and_plots(self) -> None:
        axes = _synthetic_axes()
        pred, cora, bg = _synthetic_ocean_fields(axes)

        h_slice = extract_horizontal_slice(
            pred,
            axes,
            "202401",
            "swin",
            100.0,
            cora_grid=cora,
            background_grid=bg,
            include_cora=True,
            include_anomaly=True,
        )
        z_slice = extract_zonal_section(
            pred,
            axes,
            "202401",
            "swin",
            0.0,
            cora_grid=cora,
            background_grid=bg,
            include_cora=True,
            include_anomaly=True,
        )
        m_slice = extract_meridional_section(
            pred,
            axes,
            "202401",
            "swin",
            -170.0,
            cora_grid=cora,
            background_grid=bg,
            include_cora=False,
            include_anomaly=True,
        )

        self.assertEqual(h_slice.ct.shape, (axes.latitude.size, axes.longitude.size))
        self.assertEqual(z_slice.ct.shape, (axes.depth.size, axes.longitude.size))
        self.assertEqual(m_slice.ct.shape, (axes.depth.size, axes.latitude.size))
        self.assertAlmostEqual(m_slice.fixed_coord.matched, 190.0)
        self.assertIn("CT_diff_cora", h_slice.extras)
        self.assertIn("SA_anomaly", m_slice.extras)
        self.assertTrue(np.isnan(h_slice.ct[0, 0]))  # Synthetic land corner preserved as NaN

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            for slc in (h_slice, z_slice, m_slice):
                nc_path = export_slice_netcdf(slc, out_dir / f"{slc.slug}.nc", {"test": "ok"})
                npz_path = export_slice_npz(slc, out_dir / f"{slc.slug}.npz", {"test": "ok"})
                figs = plot_ocean_slice(slc, out_dir, ("png", "pdf"))
                self.assertEqual(len(figs), 2)
                for fig_name in figs:
                    self.assertGreater((out_dir / fig_name).stat().st_size, 1000)

                # Verify NetCDF4 round-trip
                with Dataset(nc_path) as ds:
                    self.assertEqual(ds.slice_kind, slc.kind)
                    self.assertEqual(ds.month, "202401")
                    ct_read = np.ma.filled(ds["CT"][:], np.nan)
                    np.testing.assert_allclose(ct_read, slc.ct, rtol=1e-5, atol=1e-5, equal_nan=True)

                # Verify NPZ round-trip
                with np.load(npz_path, allow_pickle=False) as archive:
                    np.testing.assert_allclose(archive["SA"], slc.sa, rtol=1e-5, atol=1e-5, equal_nan=True)
                    meta = json.loads(str(archive["metadata_json"]))
                    self.assertEqual(meta["slice_kind"], slc.kind)

                summary = slice_summary_dict(slc)
                self.assertGreater(summary["CT"]["valid_cells"], 0)


if __name__ == "__main__":
    unittest.main()
