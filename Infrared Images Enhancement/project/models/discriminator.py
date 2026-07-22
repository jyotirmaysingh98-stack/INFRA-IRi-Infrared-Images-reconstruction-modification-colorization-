"""Model 3: PatchGAN discriminator for conditional GAN colorization.

Conditions on the input infrared image plus a candidate output image and
classifies overlapping patches as real/fake, encouraging locally
realistic color/texture without over-smoothing structure.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator.

    Args:
        in_channels: Combined channel count of (condition + image), e.g. 3 + 3 = 6.
        base_channels: Base channel width.
    """

    def __init__(self, in_channels: int = 6, base_channels: int = 64) -> None:
        super().__init__()

        def block(in_ch: int, out_ch: int, norm: bool = True, stride: int = 2) -> nn.Sequential:
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels, base_channels, norm=False),
            block(base_channels, base_channels * 2),
            block(base_channels * 2, base_channels * 4),
            block(base_channels * 4, base_channels * 8, stride=1),
            nn.Conv2d(base_channels * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, condition: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Classify real/fake patches given a condition and candidate image.

        Args:
            condition: Input IR image tensor (B, C, H, W).
            image: Candidate (real or generated) output image (B, C, H, W).

        Returns:
            Patch-level logits (B, 1, H', W').
        """
        x = torch.cat([condition, image], dim=1)
        return self.model(x)
