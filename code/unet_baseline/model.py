"""A compact U-Net that preserves the native odd-sized CORA grid."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConvolution(nn.Module):
    """Apply two spatial convolutions with batch-size-independent normalization."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        """Build a normalized feature extractor."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the extracted feature map."""
        return cast(torch.Tensor, self.layers(inputs))


class UNetReconstructor(nn.Module):
    """Regress CT/SA with three downsampling stages and skip connections."""

    def __init__(self, width: int = 24) -> None:
        """Configure the shallowest feature width, a multiple of eight."""
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError("width must be a positive multiple of 8")
        self.encoder1 = DoubleConvolution(276, width)
        self.encoder2 = DoubleConvolution(width, width * 2)
        self.encoder3 = DoubleConvolution(width * 2, width * 4)
        self.bottleneck = DoubleConvolution(width * 4, width * 8)
        self.decoder3 = DoubleConvolution(width * 12, width * 4)
        self.decoder2 = DoubleConvolution(width * 6, width * 2)
        self.decoder1 = DoubleConvolution(width * 3, width)
        self.head = nn.Conv2d(width, 120, 1)

    @staticmethod
    def _join(features: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Resize explicitly to the skip shape, including the 81 by 321 grid."""
        upsampled = F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((upsampled, skip), dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return normalized CT/SA anomalies as [batch, 2, 60, y, x]."""
        first = self.encoder1(inputs)
        second = self.encoder2(F.avg_pool2d(first, 2))
        third = self.encoder3(F.avg_pool2d(second, 2))
        bottom = self.bottleneck(F.avg_pool2d(third, 2))
        decoded = self.decoder3(self._join(bottom, third))
        decoded = self.decoder2(self._join(decoded, second))
        decoded = self.decoder1(self._join(decoded, first))
        result = cast(torch.Tensor, self.head(decoded))
        return result.reshape(inputs.shape[0], 2, 60, inputs.shape[-2], inputs.shape[-1])
