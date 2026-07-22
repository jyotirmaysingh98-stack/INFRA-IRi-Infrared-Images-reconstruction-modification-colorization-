"""Dataset class for infrared enhancement/colorization training.

Expects a directory layout:
    root/
        ir/   - grayscale infrared images
        rgb/  - corresponding ground-truth color images (optional; for
                 colorization training). If absent, the dataset operates
                 in "enhancement only" mode and targets are derived from
                 the classically enhanced IR image (self-supervised).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from data.preprocessing import apply_clahe, denoise_image, resize_image

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])


class IRDataset(Dataset):
    """Paired/unpaired infrared image dataset for enhancement and colorization.

    Args:
        root_dir: Root directory containing `ir/` and optionally `rgb/` subfolders.
        image_size: Target square size for resizing.
        use_clahe: Whether to pre-enhance inputs with CLAHE before feeding the model.
        denoise: Whether to denoise inputs before CLAHE.
        augment: Whether to apply training-time augmentations.
        colorize: If True, targets are RGB; otherwise targets are enhanced grayscale.
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 256,
        use_clahe: bool = True,
        denoise: bool = True,
        augment: bool = True,
        colorize: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.ir_dir = self.root_dir / "ir"
        self.rgb_dir = self.root_dir / "rgb"
        self.image_size = image_size
        self.use_clahe = use_clahe
        self.denoise = denoise
        self.colorize = colorize and self.rgb_dir.exists()

        self.ir_paths = _list_images(self.ir_dir)
        if len(self.ir_paths) == 0:
            raise FileNotFoundError(
                f"No infrared images found at {self.ir_dir}. "
                "Populate data/raw/<split>/ir/ with images."
            )

        self.transform = self._build_transform(augment, image_size)

    @staticmethod
    def _build_transform(augment: bool, image_size: int) -> A.Compose:
        """Build an albumentations transform pipeline.

        Args:
            augment: Whether to include random augmentations (train mode).
            image_size: Output square image size.

        Returns:
            Composed albumentations transform expecting `image` and `target` keys.
        """
        ops = [A.Resize(image_size, image_size)]
        if augment:
            ops += [
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(p=0.3, brightness_limit=0.15, contrast_limit=0.15),
                A.GaussNoise(p=0.15),
            ]
        ops += [A.Normalize(mean=0.5, std=0.5), ToTensorV2()]
        return A.Compose(ops, additional_targets={"target": "image"})

    def __len__(self) -> int:
        return len(self.ir_paths)

    def _load_ir(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Failed to read image: {path}")
        img = resize_image(img, (self.image_size, self.image_size))
        if self.denoise:
            img = denoise_image(img)
        if self.use_clahe:
            img = apply_clahe(img)
        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a dict with `input` and `target` tensors of shape (C, H, W) in [-1, 1].

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys `input`, `target`, and `path`.
        """
        ir_path = self.ir_paths[idx]
        ir_img = self._load_ir(ir_path)
        ir_rgb_like = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2RGB)

        if self.colorize:
            rgb_path = self.rgb_dir / ir_path.name
            target_img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if target_img is None:
                # fall back to grayscale-as-target if no paired RGB exists
                target_img = ir_rgb_like
            else:
                target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                target_img = resize_image(target_img, (self.image_size, self.image_size))
        else:
            # self-supervised enhancement target: stronger classical enhancement
            target_gray = apply_clahe(ir_img, clip_limit=3.0, tile_grid=8)
            target_img = cv2.cvtColor(target_gray, cv2.COLOR_GRAY2RGB)

        augmented = self.transform(image=ir_rgb_like, target=target_img)
        input_tensor = augmented["image"].float()
        target_tensor = augmented["target"].float()

        return {
            "input": input_tensor,
            "target": target_tensor,
            "path": str(ir_path),
        }
