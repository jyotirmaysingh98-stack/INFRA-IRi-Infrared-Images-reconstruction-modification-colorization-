# ISRO Hackathon — Problem 10: Infrared Image Colorization & Enhancement

Enhances thermal/infrared images for clearer object/boundary interpretation,
with optional colorization for human readability. Built to preserve real
structure and avoid hallucinated detail, with a deterministic inference path.

## Project Structure

```
project/
  data/
    raw/{train,val}/{ir,rgb}/   # put your images here
    dataset.py                  # PyTorch Dataset (paired IR/RGB or self-supervised)
    preprocessing.py            # resize, denoise, CLAHE, gamma, normalize
  models/
    baseline.py                 # Model 1: classical CLAHE/gamma + pseudo-color
    unet.py                     # Model 2/3: Residual U-Net enhancer/colorizer
    discriminator.py            # PatchGAN discriminator (optional GAN mode)
    losses.py                   # L1, SSIM, perceptual (VGG16), edge, adversarial
  train/
    train.py                    # training loop (supervised or GAN)
  infer/
    infer.py                    # run trained model on new images
  eval/
    evaluate.py                 # PSNR, SSIM, LPIPS, edge consistency
  app/
    app.py                      # Gradio demo
  configs/
    config.yaml                 # all hyperparameters / paths
  utils/
    config.py, seed.py, io_utils.py
  outputs/                      # logs, samples, inference results, eval CSV
  checkpoints/                  # saved model weights
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+. GPU (CUDA) is auto-detected and used if available;
falls back to CPU automatically (see `utils/config.get_device`).

## Data Preparation

Place infrared images in:
```
data/raw/train/ir/*.png
data/raw/val/ir/*.png
```

For colorization (Model 3), optionally add matching ground-truth color images
with **identical filenames**:
```
data/raw/train/rgb/*.png
data/raw/val/rgb/*.png
```

If no `rgb/` folder is provided, the dataset automatically runs in
**self-supervised enhancement mode**: the target is a stronger CLAHE
enhancement of the same image, so Model 2 (enhancer) can train without any
paired ground truth.

To enable colorization, set in `configs/config.yaml`:
```yaml
data:
  channels_out: 3
```
For grayscale-only structural enhancement, set `channels_out: 1`.

## 1) Baseline (Model 1 — no training needed)

```python
from models.baseline import BaselineEnhancer
import cv2

enhancer = BaselineEnhancer(size=(256, 256))
img = cv2.imread("sample.png", cv2.IMREAD_GRAYSCALE)
enhanced = enhancer(img)
colorized = enhancer.colorize_pseudo(enhanced)
```

This is also used inside `infer/infer.py` as a comparison reference.

## 2) Train the Deep Model (Model 2/3)

```bash
python train/train.py --config configs/config.yaml
```

- Set `train.use_gan: true` in the config to enable the conditional GAN
  (PatchGAN discriminator + adversarial loss) for colorization.
- Checkpoints saved to `checkpoints/unet_best.pt` (best val loss) and
  `checkpoints/unet_epoch{N}.pt` every `save_every` epochs.
- Sample input/output/target images saved to `outputs/samples/epoch_{N}/`.
- TensorBoard logs written to `outputs/logs/`:
  ```bash
  tensorboard --logdir outputs/logs
  ```

## 3) Run Inference on New Images

```bash
python infer/infer.py --input path/to/image.png --checkpoint checkpoints/unet_best.pt
# or a whole folder:
python infer/infer.py --input path/to/folder/
```

Outputs (in `outputs/inference/`) per image:
- `{name}_baseline.png` — classical enhancement
- `{name}_baseline_color.png` — classical pseudo-colorized
- `{name}_enhanced.png` — deep model output
- `{name}_comparison.png` — side-by-side original/baseline/deep-model

Inference is deterministic (seeded via `utils/seed.set_seed`).

## 4) Evaluate

```bash
python eval/evaluate.py --checkpoint checkpoints/unet_best.pt
```

Computes per-image and averaged **PSNR**, **SSIM**, **LPIPS**, and a
**Sobel-edge-consistency** score (boundary fidelity), saved to
`outputs/eval_results.csv`.

## 5) Demo App

```bash
python app/app.py
```

Launches a Gradio UI to upload an IR image and compare classical baseline
vs. deep-model enhancement/colorization side by side. If no checkpoint is
trained yet, the app gracefully falls back to showing the classical
baseline with a warning.

## Design Notes

- **Structure preservation**: the U-Net uses residual blocks and skip
  connections so fine boundaries from the encoder are reused in the
  decoder rather than reconstructed purely from a bottleneck, and the
  edge loss explicitly penalizes Sobel-gradient mismatch.
- **Avoiding fake detail**: Non-Local Means denoising (edge-preserving) is
  used instead of blurring; CLAHE is local-contrast only, not generative.
  Perceptual/adversarial losses are weighted low by default relative to
  L1/SSIM/edge to bias toward fidelity over plausible-looking invention.
- **Reproducibility**: `utils/seed.set_seed` fixes Python/NumPy/Torch RNGs
  and sets deterministic cuDNN flags for both training and inference.
- **Config-driven**: all paths/hyperparameters live in `configs/config.yaml`;
  no hardcoded values in the training/inference/eval code.

## Extending

- Swap in a different backbone in `models/unet.py` by changing
  `base_channels`/`depth` in the config — no other code changes needed.
- Add Weights & Biases logging by replacing/augmenting the
  `SummaryWriter` calls in `train/train.py` with `wandb.log(...)`.
- Streamlit alternative: the same `infer.preprocess_for_model` /
  `models.baseline.BaselineEnhancer` functions can be reused to build a
  Streamlit app analogous to `app/app.py`.
