"""Loss functions for IR enhancement/colorization training.

Includes pixel-wise L1, SSIM structural loss, VGG perceptual loss, a
Sobel-based edge-consistency loss (key for boundary preservation), and
standard adversarial (LSGAN) losses for the optional GAN model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16


class SSIMLoss(nn.Module):
    """Differentiable structural similarity loss (1 - SSIM).

    Args:
        window_size: Size of the Gaussian window.
        sigma: Standard deviation of the Gaussian window.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer("window", self._create_window(window_size, sigma))

    @staticmethod
    def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size: int, sigma: float) -> torch.Tensor:
        g1d = self._gaussian(window_size, sigma).unsqueeze(1)
        g2d = g1d @ g1d.t()
        return g2d.unsqueeze(0).unsqueeze(0)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute 1 - mean SSIM between two batches of images.

        Args:
            img1: Tensor (B, C, H, W) in [-1, 1] or [0, 1].
            img2: Tensor (B, C, H, W), same range as img1.

        Returns:
            Scalar loss tensor.
        """
        channels = img1.shape[1]
        window = self.window.expand(channels, 1, self.window_size, self.window_size).to(img1.device)
        pad = self.window_size // 2

        mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
        mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return 1.0 - ssim_map.mean()


class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss computed on early/mid feature maps.

    Args:
        layers: Indices into `vgg.features` to extract activations from.
    """

    def __init__(self, layers: tuple[int, ...] = (3, 8, 15)) -> None:
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.layers = layers
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual L1 distance over selected VGG feature maps.

        Args:
            pred: Predicted image (B, 3, H, W) in [-1, 1].
            target: Target image (B, 3, H, W) in [-1, 1].

        Returns:
            Scalar loss tensor.
        """
        p = self._normalize(pred)
        t = self._normalize(target)
        loss = 0.0
        x_p, x_t = p, t
        for i, layer in enumerate(self.vgg):
            x_p = layer(x_p)
            x_t = layer(x_t)
            if i in self.layers:
                loss = loss + F.l1_loss(x_p, x_t)
        return loss


class EdgeLoss(nn.Module):
    """Sobel-gradient based edge-consistency loss for boundary preservation."""

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 distance between Sobel edge maps of prediction and target.

        Args:
            pred: Predicted image (B, C, H, W).
            target: Target image (B, C, H, W).

        Returns:
            Scalar loss tensor.
        """
        return F.l1_loss(self._gradient_magnitude(pred), self._gradient_magnitude(target))


class GANLoss(nn.Module):
    """LSGAN-style adversarial loss (MSE against real/fake labels)."""

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Compute adversarial loss against a constant real/fake label.

        Args:
            prediction: Discriminator output logits.
            target_is_real: Whether the target label is "real" (1) or "fake" (0).

        Returns:
            Scalar loss tensor.
        """
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.loss(prediction, target)


class CombinedLoss(nn.Module):
    """Weighted combination of L1, SSIM, perceptual, and edge losses.

    Args:
        weights: Dict with keys "l1", "ssim", "perceptual", "edge".
        use_perceptual: Whether to include the VGG perceptual term (requires 3-channel input).
    """

    def __init__(self, weights: dict[str, float], use_perceptual: bool = True) -> None:
        super().__init__()
        self.weights = weights
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.edge = EdgeLoss()
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual = PerceptualLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute all loss terms and their weighted sum.

        Args:
            pred: Predicted image (B, C, H, W) in [-1, 1].
            target: Target image (B, C, H, W) in [-1, 1].

        Returns:
            Dict with individual loss terms and the combined "total" loss.
        """
        losses = {
            "l1": self.l1(pred, target) * self.weights.get("l1", 1.0),
            "ssim": self.ssim(pred, target) * self.weights.get("ssim", 1.0),
            "edge": self.edge(pred, target) * self.weights.get("edge", 1.0),
        }
        if self.use_perceptual and pred.shape[1] == 3:
            losses["perceptual"] = self.perceptual(pred, target) * self.weights.get("perceptual", 1.0)
        else:
            losses["perceptual"] = torch.tensor(0.0, device=pred.device)

        losses["total"] = sum(losses.values())
        return losses
