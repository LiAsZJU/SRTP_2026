#!/usr/bin/env python3
"""Sample model fields at Argo positions on the native rectilinear CORA grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset
from torch.nn import functional as F


class GridSampler:
    """Map geographic coordinates through the actual native grid centres."""

    def __init__(self, grid_path: Path, device: torch.device) -> None:
        """Read and validate the native latitude and longitude axes."""
        with Dataset(grid_path) as grid:
            latitude = np.asarray(grid["latitude"][:], dtype=np.float32)
            longitude = np.asarray(grid["longitude"][:], dtype=np.float32)
        for name, axis in (("latitude", latitude), ("longitude", longitude)):
            if axis.ndim != 1 or axis.size < 2 or not np.all(np.diff(axis) > 0) or not np.isfinite(axis).all():
                raise ValueError(f"Invalid native {name} axis in {grid_path}")
        self.latitude = torch.from_numpy(latitude).to(device)
        self.longitude = torch.from_numpy(longitude).to(device)

    @staticmethod
    def _normalized_index(axis: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Return interpolated tensor indices mapped to grid_sample coordinates."""
        positions = positions.contiguous().clamp(axis[0], axis[-1])
        upper = torch.searchsorted(axis, positions).clamp(1, axis.numel() - 1)
        lower = upper - 1
        fraction = (positions - axis[lower]) / (axis[upper] - axis[lower])
        return 2.0 * (lower + fraction) / (axis.numel() - 1) - 1.0

    def __call__(self, field: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Bilinearly sample a one-month field at [latitude, longitude, time] positions.

        Args:
            field: Tensor with shape [1, variables, depths, latitude, longitude].
            positions: Tensor with one [latitude, longitude, time] row per profile.

        Returns:
            A tensor with shape [profiles, variables, depths].
        """
        _, variables, depths, ny, nx = field.shape
        x = self._normalized_index(self.longitude, positions[:, 1])
        y = self._normalized_index(self.latitude, positions[:, 0])
        grid = torch.stack((x, y), dim=-1)
        sampled = F.grid_sample(
            field.reshape(1, variables * depths, ny, nx),
            grid[None, :, None, :],
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled[0, :, :, 0].T.reshape(-1, variables, depths)
