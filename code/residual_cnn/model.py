"""A fully convolutional residual network with dilated spatial filters."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Refine a feature map while preserving its resolution."""

    def __init__(self, width: int, dilation: int) -> None:
        """Build two normalized convolutions with an identity connection."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the residual update plus the input feature map."""
        return inputs + cast(torch.Tensor, self.layers(inputs))


class ResidualCNN(nn.Module):
    """Map the same 276 input channels as Swin to two 60-depth fields."""

    def __init__(self, width: int = 48, blocks: int = 4) -> None:
        """Configure feature width and the number of residual blocks."""
        super().__init__()
        if width < 8 or width % 8 or blocks < 1:
            raise ValueError("width must be a positive multiple of 8 and blocks must be positive")
        self.stem = nn.Conv2d(276, width, 1)
        self.blocks = nn.Sequential(*(ResidualBlock(width, 2 ** (index % 3)) for index in range(blocks)))
        self.head = nn.Conv2d(width, 120, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return normalized CT/SA anomalies as [batch, 2, 60, y, x]."""
        result = cast(torch.Tensor, self.head(self.blocks(self.stem(inputs))))
        return result.reshape(inputs.shape[0], 2, 60, inputs.shape[-2], inputs.shape[-1])
