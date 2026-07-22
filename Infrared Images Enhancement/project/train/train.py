"""Training script for the IR enhancement / colorization U-Net (Model 2/3).

Run:
    python train/train.py --config configs/config.yaml

Supports plain supervised ResUNet training and an optional conditional
GAN mode (set `train.use_gan: true` in the config) which adds the
PatchGAN discriminator and adversarial loss.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import IRDataset
from models.discriminator import PatchDiscriminator
from models.losses import CombinedLoss, GANLoss
from models.unet import build_model
from utils.config import get_device, load_config
from utils.io_utils import save_checkpoint, save_image
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Train IR enhancement/colorization model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser.parse_args()


def build_dataloaders(cfg: Dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders from config.

    Args:
        cfg: Full configuration dictionary.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    dcfg = cfg["data"]
    colorize = dcfg["channels_out"] == 3 and cfg["model"]["name"] != "enhance_only"

    train_ds = IRDataset(
        dcfg["train_dir"], dcfg["image_size"], dcfg["use_clahe"], dcfg["denoise"],
        augment=True, colorize=colorize,
    )
    val_dir = Path(dcfg["val_dir"])
    if val_dir.exists() and any((val_dir / "ir").glob("*")):
        val_ds = IRDataset(
            dcfg["val_dir"], dcfg["image_size"], dcfg["use_clahe"], dcfg["denoise"],
            augment=False, colorize=colorize,
        )
    else:
        val_ds = train_ds  # fallback for quick local testing with tiny datasets

    train_loader = DataLoader(train_ds, batch_size=dcfg["batch_size"], shuffle=True,
                               num_workers=dcfg["num_workers"], drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=dcfg["batch_size"], shuffle=False,
                             num_workers=dcfg["num_workers"])
    return train_loader, val_loader


def train(cfg: Dict[str, Any]) -> None:
    """Run the full training loop and periodically checkpoint/sample.

    Args:
        cfg: Full configuration dictionary loaded from YAML.
    """
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])
    tcfg = cfg["train"]
    mcfg = cfg["model"]
    dcfg = cfg["data"]

    train_loader, val_loader = build_dataloaders(cfg)

    generator = build_model(
        in_channels=3, out_channels=dcfg["channels_out"],
        base_channels=mcfg["base_channels"], depth=mcfg["depth"],
    ).to(device)

    use_gan = tcfg.get("use_gan", False)
    discriminator = None
    if use_gan:
        discriminator = PatchDiscriminator(in_channels=3 + dcfg["channels_out"]).to(device)
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=tcfg["lr"],
                                  betas=(tcfg["beta1"], tcfg["beta2"]))
        gan_loss_fn = GANLoss().to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=tcfg["lr"],
                              betas=(tcfg["beta1"], tcfg["beta2"]), weight_decay=tcfg["weight_decay"])

    criterion = CombinedLoss(tcfg["loss_weights"], use_perceptual=(dcfg["channels_out"] == 3)).to(device)

    log_dir = Path(tcfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    ckpt_dir = Path(tcfg["checkpoint_dir"])
    sample_dir = Path(tcfg["sample_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, tcfg["epochs"] + 1):
        generator.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{tcfg['epochs']}")

        for batch in pbar:
            inp = batch["input"].to(device)
            target = batch["target"].to(device)

            if use_gan:
                # --- Discriminator step ---
                with torch.no_grad():
                    fake = generator(inp)
                opt_d.zero_grad()
                pred_real = discriminator(inp, target)
                pred_fake = discriminator(inp, fake.detach())
                d_loss = 0.5 * (gan_loss_fn(pred_real, True) + gan_loss_fn(pred_fake, False))
                d_loss.backward()
                opt_d.step()

            # --- Generator step ---
            opt_g.zero_grad()
            fake = generator(inp)
            loss_dict = criterion(fake, target)
            g_loss = loss_dict["total"]

            if use_gan:
                pred_fake_for_g = discriminator(inp, fake)
                adv_loss = gan_loss_fn(pred_fake_for_g, True) * tcfg["loss_weights"].get("adversarial", 0.1)
                g_loss = g_loss + adv_loss
                loss_dict["adversarial"] = adv_loss

            g_loss.backward()
            opt_g.step()

            epoch_loss += g_loss.item()
            global_step += 1
            pbar.set_postfix(loss=g_loss.item())

            for name, val in loss_dict.items():
                writer.add_scalar(f"train/{name}", val.item() if torch.is_tensor(val) else val, global_step)

        avg_train_loss = epoch_loss / len(train_loader)

        # --- Validation ---
        val_loss = validate(generator, val_loader, criterion, device)
        writer.add_scalar("val/loss", val_loss, epoch)
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                {"epoch": epoch, "model_state": generator.state_dict(), "val_loss": val_loss, "config": cfg},
                ckpt_dir / "unet_best.pt",
            )

        if epoch % tcfg["save_every"] == 0:
            save_checkpoint(
                {"epoch": epoch, "model_state": generator.state_dict(), "val_loss": val_loss, "config": cfg},
                ckpt_dir / f"unet_epoch{epoch}.pt",
            )
            save_sample_outputs(generator, val_loader, device, sample_dir / f"epoch_{epoch}")

    writer.close()


def validate(model: torch.nn.Module, loader: DataLoader, criterion: CombinedLoss, device: str) -> float:
    """Compute average validation loss over a dataloader.

    Args:
        model: Generator model to evaluate.
        loader: Validation dataloader.
        criterion: Combined loss module.
        device: Torch device string.

    Returns:
        Average validation loss as a float.
    """
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            inp = batch["input"].to(device)
            target = batch["target"].to(device)
            pred = model(inp)
            loss_dict = criterion(pred, target)
            total += loss_dict["total"].item()
    return total / max(len(loader), 1)


def save_sample_outputs(model: torch.nn.Module, loader: DataLoader, device: str, out_dir: Path,
                         num_samples: int = 4) -> None:
    """Save a few side-by-side input/output samples for visual monitoring.

    Args:
        model: Generator model.
        loader: Dataloader to draw samples from.
        device: Torch device string.
        out_dir: Output directory for saved images.
        num_samples: Number of samples to save.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            inp = batch["input"].to(device)
            pred = model(inp)
            for i in range(inp.shape[0]):
                if saved >= num_samples:
                    return
                save_image(inp[i], out_dir / f"sample{saved}_input.png")
                save_image(pred[i], out_dir / f"sample{saved}_output.png")
                save_image(batch["target"][i], out_dir / f"sample{saved}_target.png")
                saved += 1


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config)
