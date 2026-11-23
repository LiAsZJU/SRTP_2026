"""Store a fitted vertical EOF basis and a pointwise ridge mapping."""

from __future__ import annotations

import torch
from torch import nn


class EOFRidge(nn.Module):
    """Predict vertical-mode amplitudes from the 276 local input channels.

    The fit uses complete training-month CT/SA columns only. Missing input
    observations remain zero-filled with their explicit mask channels.
    """

    input_mean: torch.Tensor
    input_scale: torch.Tensor
    target_mean: torch.Tensor
    basis: torch.Tensor
    coefficients: torch.Tensor

    def __init__(self, modes: int = 16) -> None:
        """Allocate checkpoint buffers for a joint 120-dimensional vertical EOF."""
        super().__init__()
        if not 1 <= modes <= 120:
            raise ValueError("modes must be between 1 and 120")
        self.register_buffer("input_mean", torch.zeros(276))
        self.register_buffer("input_scale", torch.ones(276))
        self.register_buffer("target_mean", torch.zeros(120))
        self.register_buffer("basis", torch.zeros(120, modes))
        self.register_buffer("coefficients", torch.zeros(276, modes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reconstruct normalized CT/SA columns on the native horizontal grid."""
        rows = inputs.permute(0, 2, 3, 1)
        amplitudes = ((rows - self.input_mean) / self.input_scale) @ self.coefficients
        columns = amplitudes @ self.basis.T + self.target_mean
        fields = columns.permute(0, 3, 1, 2)
        return fields.reshape(inputs.shape[0], 2, 60, inputs.shape[-2], inputs.shape[-1])
