"""Classical preprocessing utilities for infrared images.

Includes resizing, denoising, CLAHE-based contrast enhancement, gamma
correction, and normalization helpers shared by the dataset, baseline
enhancer, and inference pipeline.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize an image to (H, W) using area interpolation (good for downsizing).

    Args:
        img: Input image (H, W) or (H, W, C).
        size: Target (height, width).

    Returns:
        Resized image.
    """
    h, w = size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def denoise_image(img: np.ndarray) -> np.ndarray:
    """Apply edge-preserving denoising to reduce thermal sensor noise.

    Uses Non-Local Means for grayscale images, which preserves edges
    better than Gaussian blur and avoids hallucinating fake detail.

    Args:
        img: uint8 grayscale or color image.

    Returns:
        Denoised uint8 image.
    """
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, None, h=7, templateWindowSize=7, searchWindowSize=21)
    return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization.

    CLAHE improves local contrast for object/boundary separation without
    over-amplifying noise globally, making it well suited for thermal data.

    Args:
        img: uint8 grayscale image (H, W).
        clip_limit: Contrast clipping threshold.
        tile_grid: Size of the grid for local histogram equalization.

    Returns:
        Contrast-enhanced uint8 image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(img)


def gamma_correction(img: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Apply gamma correction to brighten/darken mid-tones.

    Args:
        img: uint8 image.
        gamma: Gamma value; >1 brightens, <1 darkens.

    Returns:
        Gamma-corrected uint8 image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def normalize_to_unit(img: np.ndarray) -> np.ndarray:
    """Scale a uint8 image to float32 in [0, 1].

    Args:
        img: uint8 image.

    Returns:
        float32 image in [0, 1].
    """
    return img.astype(np.float32) / 255.0


def classical_pipeline(
    img: np.ndarray,
    size: Tuple[int, int] = (256, 256),
    clip_limit: float = 2.0,
    tile_grid: int = 8,
    denoise: bool = True,
    gamma: float = 1.0,
) -> np.ndarray:
    """Full classical preprocessing/enhancement pipeline (the Model 1 baseline).

    Steps: resize -> denoise -> CLAHE -> gamma correction.

    Args:
        img: uint8 grayscale infrared image.
        size: Target (height, width).
        clip_limit: CLAHE clip limit.
        tile_grid: CLAHE tile grid size.
        denoise: Whether to apply Non-Local Means denoising.
        gamma: Gamma correction value (1.0 = no-op).

    Returns:
        Enhanced uint8 grayscale image.
    """
    out = resize_image(img, size)
    if denoise:
        out = denoise_image(out)
    out = apply_clahe(out, clip_limit, tile_grid)
    if gamma != 1.0:
        out = gamma_correction(out, gamma)
    return out
