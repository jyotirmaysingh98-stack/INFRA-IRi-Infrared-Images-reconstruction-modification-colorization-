"""Model 1: Classical baseline enhancer.

Non-learned reference pipeline used to benchmark the deep models against
(histogram equalization / CLAHE / gamma correction). No training required.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from data.preprocessing import apply_clahe, denoise_image, gamma_correction, resize_image


class BaselineEnhancer:
    """Classical CLAHE + gamma correction infrared enhancer.

    Args:
        size: Output (height, width).
        clip_limit: CLAHE clip limit.
        tile_grid: CLAHE tile grid size.
        gamma: Gamma correction factor.
        denoise: Whether to denoise before enhancement.
    """

    def __init__(
        self,
        size: Tuple[int, int] = (256, 256),
        clip_limit: float = 2.0,
        tile_grid: int = 8,
        gamma: float = 1.1,
        denoise: bool = True,
    ) -> None:
        self.size = size
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid
        self.gamma = gamma
        self.denoise = denoise

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Enhance a single grayscale infrared image.

        Args:
            img: uint8 grayscale image.

        Returns:
            Enhanced uint8 grayscale image, resized to `self.size`.
        """
        out = resize_image(img, self.size)
        if self.denoise:
            out = denoise_image(out)
        out = apply_clahe(out, self.clip_limit, self.tile_grid)
        out = gamma_correction(out, self.gamma)
        return out

    def colorize_pseudo(self, img: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
        """Apply a pseudo-color map to an enhanced grayscale image for human readability.

        This is a non-learned colorization fallback (Model 1 equivalent of Model 3).

        Args:
            img: uint8 grayscale image (already enhanced).
            colormap: OpenCV colormap constant.

        Returns:
            BGR uint8 pseudo-colored image.
        """
        return cv2.applyColorMap(img, colormap)
