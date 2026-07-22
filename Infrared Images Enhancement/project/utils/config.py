"""Configuration loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a nested dictionary.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed configuration as a dictionary.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_device(preferred: str = "cuda") -> str:
    """Return the best available torch device string.

    Args:
        preferred: Preferred device ("cuda" or "cpu").

    Returns:
        "cuda" if available and preferred, else "cpu".
    """
    import torch

    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
