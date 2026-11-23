"""Represent the existing training-month climatology in normalized units."""

from __future__ import annotations

import torch
from torch import nn


class ClimatologyBaseline(nn.Module):
    """Return zero physical anomaly relative to the monthly background.

    The reader has already subtracted the calendar-month climatology. Its
    normalized representation must be ``-mean / std``, rather than zero.
    """

    normalized_background: torch.Tensor

    def __init__(self, normalized_background: torch.Tensor | None = None) -> None:
        """Register a constant field with shape [1, 2, 60, 1, 1]."""
        super().__init__()
        value = torch.zeros(1, 2, 60, 1, 1) if normalized_background is None else normalized_background
        self.register_buffer("normalized_background", value.detach().clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Expand the constant to the input batch and horizontal grid."""
        return self.normalized_background.expand(inputs.shape[0], -1, -1, inputs.shape[-2], inputs.shape[-1])
