"""Evaluation script: computes PSNR, SSIM, LPIPS, and edge consistency.

Run:
    python eval/evaluate.py --config configs/config.yaml --checkpoint checkpoints/unet_best.pt

Compares model outputs against ground-truth targets (or, if unavailable,
against the classical baseline) on the validation set and writes a CSV
of per-image and averaged metrics.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import IRDataset
from models.unet import build_model
from utils.config import get_device, load_config
from utils.io_utils import load_checkpoint, tensor_to_image
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Evaluate IR enhancement/colorization model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def edge_consistency(pred_gray: np.ndarray, target_gray: np.ndarray) -> float:
    """Compute edge-map correlation between prediction and target (boundary fidelity).

    Args:
        pred_gray: uint8 grayscale prediction.
        target_gray: uint8 grayscale target.

    Returns:
        Normalized cross-correlation of Canny edge maps in [0, 1] (higher = better).
    """
    edges_pred = cv2.Canny(pred_gray, 100, 200).astype(np.float32)
    edges_target = cv2.Canny(target_gray, 100, 200).astype(np.float32)
    num = np.sum(edges_pred * edges_target)
    den = np.sqrt(np.sum(edges_pred ** 2) * np.sum(edges_target ** 2)) + 1e-8
    return float(num / den)


def evaluate(cfg: Dict[str, Any], checkpoint_path: str) -> None:
    """Run full evaluation over the validation set and save a results CSV.

    Args:
        cfg: Full configuration dictionary.
        checkpoint_path: Path to the trained model checkpoint.
    """
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])
    dcfg = cfg["data"]
    mcfg = cfg["model"]

    colorize = dcfg["channels_out"] == 3
    val_ds = IRDataset(dcfg["val_dir"], dcfg["image_size"], dcfg["use_clahe"], dcfg["denoise"],
                        augment=False, colorize=colorize)

    model = build_model(in_channels=3, out_channels=dcfg["channels_out"],
                         base_channels=mcfg["base_channels"], depth=mcfg["depth"]).to(device)
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    lpips_fn = lpips.LPIPS(net="alex").to(device)

    rows: List[Dict[str, Any]] = []
    psnr_vals, ssim_vals, lpips_vals, edge_vals = [], [], [], []

    with torch.no_grad():
        for idx in range(len(val_ds)):
            sample = val_ds[idx]
            inp = sample["input"].unsqueeze(0).to(device)
            target = sample["target"].unsqueeze(0).to(device)
            pred = model(inp)

            pred_img = tensor_to_image(pred[0])
            target_img = tensor_to_image(target[0])

            pred_gray = pred_img if pred_img.ndim == 2 else cv2.cvtColor(pred_img, cv2.COLOR_RGB2GRAY)
            target_gray = target_img if target_img.ndim == 2 else cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)

            psnr = sk_psnr(target_gray, pred_gray, data_range=255)
            ssim = sk_ssim(target_gray, pred_gray, data_range=255)
            lpips_val = lpips_fn(pred, target).item()
            edge_val = edge_consistency(pred_gray, target_gray)

            psnr_vals.append(psnr)
            ssim_vals.append(ssim)
            lpips_vals.append(lpips_val)
            edge_vals.append(edge_val)

            rows.append({
                "image": Path(sample["path"]).name,
                "psnr": psnr, "ssim": ssim, "lpips": lpips_val, "edge_consistency": edge_val,
            })

    out_path = Path(cfg["eval"]["results_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "psnr", "ssim", "lpips", "edge_consistency"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Evaluated {len(rows)} images. Results saved to {out_path}")
    print(f"Mean PSNR: {np.mean(psnr_vals):.3f}")
    print(f"Mean SSIM: {np.mean(ssim_vals):.4f}")
    print(f"Mean LPIPS: {np.mean(lpips_vals):.4f} (lower is better)")
    print(f"Mean Edge Consistency: {np.mean(edge_vals):.4f}")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    ckpt_path = args.checkpoint or config["eval"]["checkpoint_path"]
    evaluate(config, ckpt_path)
