"""Inference script: run the trained model on new infrared images.

Run:
    python infer/infer.py --input path/to/image_or_dir --config configs/config.yaml \
        --checkpoint checkpoints/unet_best.pt

Produces, for each input image: the classical-baseline enhancement, the
deep-model enhanced/colorized output, and a side-by-side comparison.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.preprocessing import apply_clahe, denoise_image, resize_image
from models.baseline import BaselineEnhancer
from models.unet import build_model
from utils.config import get_device, load_config
from utils.io_utils import load_checkpoint, tensor_to_image
from utils.seed import set_seed

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Run inference on infrared images")
    parser.add_argument("--input", type=str, required=True, help="Image file or directory")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def load_model(cfg: dict, checkpoint_path: str, device: str) -> torch.nn.Module:
    """Load a trained generator model from a checkpoint.

    Args:
        cfg: Full configuration dictionary.
        checkpoint_path: Path to the .pt checkpoint file.
        device: Torch device string.

    Returns:
        Model in eval mode, loaded with trained weights.
    """
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    model = build_model(
        in_channels=3, out_channels=dcfg["channels_out"],
        base_channels=mcfg["base_channels"], depth=mcfg["depth"],
    ).to(device)
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def preprocess_for_model(img_gray: np.ndarray, cfg: dict) -> torch.Tensor:
    """Replicate the dataset preprocessing for a single raw grayscale image.

    Args:
        img_gray: uint8 grayscale image.
        cfg: Full configuration dictionary.

    Returns:
        Normalized input tensor (1, 3, H, W) in [-1, 1].
    """
    dcfg = cfg["data"]
    size = dcfg["image_size"]
    out = resize_image(img_gray, (size, size))
    if dcfg["denoise"]:
        out = denoise_image(out)
    if dcfg["use_clahe"]:
        out = apply_clahe(out, dcfg["clahe_clip_limit"], dcfg["clahe_tile_grid"])
    rgb_like = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    rgb_like = (rgb_like - 0.5) / 0.5
    tensor = torch.from_numpy(rgb_like).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def make_side_by_side(images: List[np.ndarray]) -> np.ndarray:
    """Stack a list of equally-sized images horizontally for comparison.

    Args:
        images: List of HxW or HxWx3 uint8 images.

    Returns:
        Single horizontally concatenated uint8 image.
    """
    normalized = []
    for img in images:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        normalized.append(img)
    return np.concatenate(normalized, axis=1)


def run_inference(image_path: Path, model: torch.nn.Module, baseline: BaselineEnhancer,
                   cfg: dict, device: str, out_dir: Path) -> None:
    """Run baseline + deep model inference on a single image and save outputs.

    Args:
        image_path: Path to the input infrared image.
        model: Loaded deep learning model.
        baseline: Classical baseline enhancer instance.
        cfg: Full configuration dictionary.
        device: Torch device string.
        out_dir: Directory to write outputs into.
    """
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Skipping unreadable file: {image_path}")
        return

    baseline_out = baseline(img_gray)
    baseline_color = baseline.colorize_pseudo(baseline_out)

    inp_tensor = preprocess_for_model(img_gray, cfg).to(device)
    with torch.no_grad():
        pred = model(inp_tensor)[0]
    deep_out = tensor_to_image(pred)
    if deep_out.ndim == 2:
        deep_out_bgr = cv2.cvtColor(deep_out, cv2.COLOR_GRAY2BGR)
    else:
        deep_out_bgr = cv2.cvtColor(deep_out, cv2.COLOR_RGB2BGR)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    cv2.imwrite(str(out_dir / f"{stem}_baseline.png"), baseline_out)
    cv2.imwrite(str(out_dir / f"{stem}_baseline_color.png"), baseline_color)
    cv2.imwrite(str(out_dir / f"{stem}_enhanced.png"), deep_out_bgr)

    size = cfg["data"]["image_size"]
    orig_resized = cv2.cvtColor(resize_image(img_gray, (size, size)), cv2.COLOR_GRAY2BGR)
    baseline_resized = cv2.cvtColor(resize_image(baseline_out, (size, size)), cv2.COLOR_GRAY2BGR)
    comparison = make_side_by_side([orig_resized, baseline_resized, deep_out_bgr])
    cv2.imwrite(str(out_dir / f"{stem}_comparison.png"), comparison)
    print(f"Saved outputs for {image_path.name} -> {out_dir}")


def main() -> None:
    """Entry point: load config/model and run inference over input path."""
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])

    checkpoint_path = args.checkpoint or cfg["infer"]["checkpoint_path"]
    model = load_model(cfg, checkpoint_path, device)
    baseline = BaselineEnhancer(
        size=(cfg["data"]["image_size"], cfg["data"]["image_size"]),
        clip_limit=cfg["data"]["clahe_clip_limit"],
        tile_grid=cfg["data"]["clahe_tile_grid"],
    )

    input_path = Path(args.input)
    out_dir = Path(cfg["infer"]["output_dir"])

    if input_path.is_dir():
        paths = sorted([p for p in input_path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])
    else:
        paths = [input_path]

    for p in paths:
        run_inference(p, model, baseline, cfg, device, out_dir)


if __name__ == "__main__":
    main()
