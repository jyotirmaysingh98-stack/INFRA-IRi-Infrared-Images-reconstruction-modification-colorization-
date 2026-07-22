"""Model 2/3: Residual U-Net for IR enhancement and colorization.

A single configurable architecture: set `out_channels=1` for structure-
preserving enhancement, or `out_channels=3` for colorization. Residual
blocks help preserve fine structural detail and avoid hallucinated
artifacts.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two conv layers with a residual skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class DownBlock(nn.Module):
    """Conv -> InstanceNorm -> ReLU -> Residual -> downsample."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ResidualBlock(out_ch),
        )
        self.pool = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        down = self.pool(feat)
        return feat, down


class UpBlock(nn.Module):
    """Upsample -> concat skip -> conv -> residual."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ResidualBlock(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResUNet(nn.Module):
    """Residual U-Net for infrared enhancement / colorization.

    Args:
        in_channels: Number of input channels (3 if IR replicated to RGB-like).
        out_channels: Number of output channels (1 = enhanced grayscale, 3 = color).
        base_channels: Number of channels at the first encoder level.
        depth: Number of down/up sampling levels.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.depth = depth
        chs: List[int] = [base_channels * (2 ** i) for i in range(depth)]

        self.downs = nn.ModuleList()
        prev_ch = in_channels
        for ch in chs:
            self.downs.append(DownBlock(prev_ch, ch))
            prev_ch = ch

        self.bottleneck = nn.Sequential(
            nn.Conv2d(prev_ch, prev_ch * 2, 3, padding=1),
            nn.InstanceNorm2d(prev_ch * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(prev_ch * 2),
            ResidualBlock(prev_ch * 2),
        )

        self.ups = nn.ModuleList()
        up_in = prev_ch * 2
        for ch in reversed(chs):
            self.ups.append(UpBlock(up_in, ch, ch))
            up_in = ch

        self.out_conv = nn.Sequential(
            nn.Conv2d(up_in, out_channels, 3, padding=1),
            nn.Tanh(),  # output in [-1, 1] to match normalized targets
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, in_channels, H, W).

        Returns:
            Output tensor (B, out_channels, H, W) in [-1, 1].
        """
        skips = []
        out = x
        for down in self.downs:
            feat, out = down(out)
            skips.append(feat)

        out = self.bottleneck(out)

        for up, skip in zip(self.ups, reversed(skips)):
            out = up(out, skip)

        return self.out_conv(out)


def build_model(
    in_channels: int = 3,
    out_channels: int = 3,
    base_channels: int = 64,
    depth: int = 4,
) -> ResUNet:
    """Factory function to build a ResUNet model.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        base_channels: Base channel width.
        depth: Encoder/decoder depth.

    Returns:
        Instantiated ResUNet model.
    """
    return ResUNet(in_channels, out_channels, base_channels, depth)
