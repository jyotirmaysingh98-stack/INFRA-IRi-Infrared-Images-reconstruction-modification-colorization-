"""Gradio demo app for infrared image enhancement and colorization.

Run:
    python app/app.py

Lets a user upload an infrared image and view: the classical baseline
enhancement, the deep-learning enhanced/colorized output, and a
side-by-side comparison, with an optional pseudo-color overlay.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from infer.infer import load_model, preprocess_for_model
from models.baseline import BaselineEnhancer
from utils.config import get_device, load_config
from utils.io_utils import load_checkpoint, tensor_to_image
from utils.seed import set_seed

CONFIG_PATH = "configs/config.yaml"

cfg = load_config(CONFIG_PATH)
set_seed(cfg["seed"])
device = get_device(cfg["train"]["device"])

_model_cache = {"model": None}


def get_model() -> torch.nn.Module | None:
    """Lazily load and cache the trained model, returning None if unavailable.

    Returns:
        Loaded model or None if no checkpoint exists yet.
    """
    if _model_cache["model"] is not None:
        return _model_cache["model"]
    ckpt_path = Path(cfg["infer"]["checkpoint_path"])
    if not ckpt_path.exists():
        return None
    try:
        model = load_model(cfg, str(ckpt_path), device)
        _model_cache["model"] = model
        return model
    except Exception as e:  # noqa: BLE001
        print(f"Could not load checkpoint: {e}")
        return None


baseline = BaselineEnhancer(
    size=(cfg["data"]["image_size"], cfg["data"]["image_size"]),
    clip_limit=cfg["data"]["clahe_clip_limit"],
    tile_grid=cfg["data"]["clahe_tile_grid"],
)


def process(image: np.ndarray, colormap_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process an uploaded image through baseline and deep model pipelines.

    Args:
        image: Input image as a numpy array (RGB, from Gradio).
        colormap_name: Name of the OpenCV colormap to use for the baseline pseudo-color view.

    Returns:
        Tuple of (baseline_enhanced, baseline_colorized, deep_model_output) as RGB arrays.
    """
    if image is None:
        raise gr.Error("Please upload an infrared image first.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    baseline_out = baseline(gray)
    colormap = getattr(cv2, f"COLORMAP_{colormap_name.upper()}", cv2.COLORMAP_INFERNO)
    baseline_color_bgr = baseline.colorize_pseudo(baseline_out, colormap)
    baseline_color_rgb = cv2.cvtColor(baseline_color_bgr, cv2.COLOR_BGR2RGB)

    model = get_model()
    if model is None:
        deep_out_rgb = cv2.cvtColor(baseline_out, cv2.COLOR_GRAY2RGB)
        gr.Warning("No trained checkpoint found yet — showing classical baseline as placeholder. "
                    "Train the model first (train/train.py) for deep-learning output.")
    else:
        inp_tensor = preprocess_for_model(gray, cfg).to(device)
        with torch.no_grad():
            pred = model(inp_tensor)[0]
        deep_out = tensor_to_image(pred)
        deep_out_rgb = deep_out if deep_out.ndim == 3 else cv2.cvtColor(deep_out, cv2.COLOR_GRAY2RGB)

    return cv2.cvtColor(baseline_out, cv2.COLOR_GRAY2RGB), baseline_color_rgb, deep_out_rgb


def build_demo() -> gr.Blocks:
    """Build the Gradio Blocks interface.

    Returns:
        Configured Gradio Blocks app.
    """
    with gr.Blocks(title="ISRO IR Image Enhancement & Colorization") as demo:
        gr.Markdown(
            "# Infrared Image Colorization & Enhancement\n"
            "Upload a thermal/infrared image to compare the classical baseline "
            "(CLAHE + gamma + pseudo-color) against the deep learning enhancer/colorizer."
        )
        with gr.Row():
            inp_image = gr.Image(label="Infrared Input", type="numpy")
            colormap = gr.Dropdown(
                choices=["INFERNO", "JET", "HOT", "VIRIDIS", "MAGMA", "PLASMA"],
                value="INFERNO", label="Baseline Pseudo-color Map",
            )
        run_btn = gr.Button("Enhance & Colorize", variant="primary")
        with gr.Row():
            out_baseline = gr.Image(label="Classical Baseline (Enhanced Grayscale)")
            out_baseline_color = gr.Image(label="Classical Baseline (Pseudo-colorized)")
            out_deep = gr.Image(label="Deep Model Output")

        run_btn.click(process, inputs=[inp_image, colormap],
                       outputs=[out_baseline, out_baseline_color, out_deep])

        gr.Markdown(
            "Note: structure preservation is prioritized — outputs are designed to "
            "enhance real object boundaries rather than hallucinate new detail."
        )
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=True)
