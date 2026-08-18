"""
HANSR Metrics — PSNR, SSIM, LPIPS Wrappers

Evaluation metrics for image restoration quality assessment.

Note on LPIPS: The backbone is pretrained on RGB ImageNet. For grayscale,
we replicate the single channel to 3 channels. LPIPS values should be
treated as comparative-only between HANSR configs, not absolute.
"""

import logging
from typing import Dict

import torch
import numpy as np

logger = logging.getLogger("hansr")


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted image tensor.
        gt: Ground truth tensor (same shape).
        max_val: Maximum pixel value (1.0 for normalized images).

    Returns:
        PSNR in dB (higher = better).
    """
    mse = torch.mean((pred - gt) ** 2).item()
    if mse < 1e-10:
        return 100.0  # effectively identical
    return 10.0 * np.log10(max_val ** 2 / mse)


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Structural Similarity Index (via scikit-image).

    Args:
        pred: Predicted image tensor (B, 1, H, W) or (1, H, W).
        gt: Ground truth tensor (same shape).

    Returns:
        SSIM value in [0, 1] (higher = better).
    """
    from skimage.metrics import structural_similarity as ssim

    p = pred.detach().cpu().squeeze().numpy()
    g = gt.detach().cpu().squeeze().numpy()

    # Handle batch dimension
    if p.ndim == 3:  # (B, H, W)
        scores = [ssim(p[i], g[i], data_range=1.0) for i in range(p.shape[0])]
        return float(np.mean(scores))
    else:
        return float(ssim(p, g, data_range=1.0))


# LPIPS singleton (lazy-loaded to avoid slow import at module level)
_lpips_model = None


def _get_lpips_model(device: torch.device):
    """Lazy-load LPIPS model (singleton)."""
    global _lpips_model
    if _lpips_model is None:
        try:
            import lpips
            _lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device)
            _lpips_model.eval()
            logger.info("Loaded LPIPS model (AlexNet backbone)")
        except Exception:
            _lpips_model = "UNAVAILABLE"
    return _lpips_model


def compute_lpips(
    pred: torch.Tensor, gt: torch.Tensor, device: torch.device = None
) -> float:
    """
    Learned Perceptual Image Patch Similarity.

    NOTE: LPIPS is pretrained on RGB. For grayscale, we replicate to 3ch.
    Values are comparative-only between HANSR configs, not absolute.

    Args:
        pred: (B, 1, H, W) or (1, H, W) tensor.
        gt: Same shape as pred.
        device: Device for LPIPS model.

    Returns:
        LPIPS distance (lower = better / more similar).
    """
    if device is None:
        device = pred.device

    model = _get_lpips_model(device)
    if model == "UNAVAILABLE":
        return 0.0

    # Ensure 4D
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    # Replicate grayscale to 3-channel for LPIPS
    pred_rgb = pred.repeat(1, 3, 1, 1).to(device)
    gt_rgb = gt.repeat(1, 3, 1, 1).to(device)

    # LPIPS expects [-1, 1]; our images are [0, 1]
    pred_rgb = pred_rgb * 2.0 - 1.0
    gt_rgb = gt_rgb * 2.0 - 1.0

    with torch.no_grad():
        score = model(pred_rgb, gt_rgb)

    return float(score.mean().item())


def compute_all_metrics(
    pred: torch.Tensor, gt: torch.Tensor, device: torch.device = None
) -> Dict[str, float]:
    """Compute all three metrics at once."""
    return {
        "psnr": compute_psnr(pred, gt),
        "ssim": compute_ssim(pred, gt),
        "lpips": compute_lpips(pred, gt, device=device),
    }
