"""I/O helpers for checkpoints and image saving."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    """Save a training checkpoint to disk.

    Args:
        state: Dictionary containing model/optimizer state and metadata.
        path: Destination file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a training checkpoint from disk.

    Args:
        path: Path to checkpoint file.
        map_location: Device to map tensors to.

    Returns:
        The loaded checkpoint dictionary.
    """
    return torch.load(path, map_location=map_location)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW float tensor in [-1, 1] or [0, 1] to an HWC uint8 numpy image.

    Args:
        tensor: Image tensor of shape (C, H, W).

    Returns:
        uint8 numpy array of shape (H, W, C) or (H, W) for single channel.
    """
    img = tensor.detach().cpu().float()
    if img.min() < 0:
        img = (img + 1.0) / 2.0
    img = img.clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0))
    if img.shape[2] == 1:
        img = img[:, :, 0]
    return (img * 255.0).astype(np.uint8)


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a tensor as an image file.

    Args:
        tensor: Image tensor of shape (C, H, W).
        path: Destination file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = tensor_to_image(tensor)
    Image.fromarray(arr).save(path)
