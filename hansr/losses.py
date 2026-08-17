"""
HANSR Loss Functions — Five-Term Composite Loss (FR-007)

Each term is individually removable for ablation. All weights are config-driven.

Terms:
  1. Charbonnier  — robust pixel-level reconstruction (L1 variant)
  2. Edge (Sobel) — preserve edge structure
  3. FFT Magnitude — anti-hallucination: penalize unsupported high-freq energy (FR-006)
  4. Range Penalty — soft [0,1] output range enforcement
  5. Total Variation — mild smoothness regularization

Reference: PRD Section 9 (FR-007), Section 3 (FR-006 anti-hallucination).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# =============================================================================
# Individual Loss Terms
# =============================================================================

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss — robust L1 variant: sqrt(||pred - gt||^2 + eps^2).

    More robust to outliers than L2 while remaining differentiable at zero
    (unlike raw L1). Standard choice for image restoration.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps_sq = eps ** 2

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        diff = pred - gt
        return torch.mean(torch.sqrt(diff * diff + self.eps_sq))


class EdgeLoss(nn.Module):
    """
    Sobel Edge Loss — Charbonnier loss on Sobel-filtered edge maps.

    Penalizes edge distortion between predicted and ground-truth images.
    Uses fixed (non-learnable) 3x3 Sobel kernels for horizontal and vertical
    gradient extraction.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps_sq = eps ** 2

        # Sobel kernels (fixed, not learned)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _sobel_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Extract edge magnitude from single-channel image."""
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred_edges = self._sobel_edges(pred)
        gt_edges = self._sobel_edges(gt)
        diff = pred_edges - gt_edges
        return torch.mean(torch.sqrt(diff * diff + self.eps_sq))


class FFTLoss(nn.Module):
    """
    FFT Magnitude Loss — L1 distance in frequency domain.

    Anti-hallucination constraint (FR-006): penalizes the model for producing
    high-frequency energy that doesn't exist in the ground truth. This prevents
    invented texture, sharpening halos, and ringing artifacts — critical for
    inspection-adjacent restoration.
    """

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # 2D FFT on spatial dimensions
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        gt_fft = torch.fft.rfft2(gt, norm="ortho")

        # Compare magnitudes (phase differences are less meaningful for this task)
        pred_mag = torch.abs(pred_fft)
        gt_mag = torch.abs(gt_fft)

        return F.l1_loss(pred_mag, gt_mag)


class RangePenaltyLoss(nn.Module):
    """
    Soft Range Penalty — penalizes output values outside [0, 1].

    Uses squared penalty: mean(max(0, pred-1)^2 + max(0, -pred)^2).
    This is a soft constraint, not a hard clamp — the network can still
    output values outside [0,1] during training, but is gently discouraged.

    Note: Input degraded values may legitimately exceed [0,1] (FR-004) — this
    penalty applies only to the *output*, not the input.
    """

    def forward(self, pred: torch.Tensor, gt: torch.Tensor = None) -> torch.Tensor:
        # gt is unused but accepted for uniform interface
        over = F.relu(pred - 1.0)
        under = F.relu(-pred)
        return torch.mean(over * over + under * under)


class TVLoss(nn.Module):
    """
    Total Variation Loss — mild smoothness regularization.

    Penalizes high-frequency noise in the output by computing the L1 norm
    of spatial gradients: ||nabla_x pred||_1 + ||nabla_y pred||_1.

    Weight should be kept low to avoid over-smoothing detail.
    """

    def forward(self, pred: torch.Tensor, gt: torch.Tensor = None) -> torch.Tensor:
        # gt is unused but accepted for uniform interface
        # Finite differences along H and W dimensions
        diff_h = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        diff_w = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()


# =============================================================================
# Composite Loss — Config-Driven Weighted Sum (FR-007)
# =============================================================================

class CompositeLoss(nn.Module):
    """
    Five-term composite loss with config-driven weights.

    Each term can be individually disabled by setting its weight to 0 (or
    removing it from config), enabling clean single-variable ablation (FR-016).

    Args:
        config: Loss section of the YAML config. Expected structure:
            loss:
              charbonnier: {weight: 1.0, eps: 1e-3}
              edge: {weight: 0.1}
              fft: {weight: 0.05}
              range_penalty: {weight: 0.01}
              tv: {weight: 0.001}
    """

    def __init__(self, config: dict):
        super().__init__()
        loss_cfg = config.get("loss", config)  # accept either full config or loss section

        # Build active loss terms based on config weights
        self.terms = nn.ModuleDict()
        self.weights = {}

        # Charbonnier
        charb_cfg = loss_cfg.get("charbonnier", {})
        w = charb_cfg.get("weight", 1.0)
        if w > 0:
            eps = charb_cfg.get("eps", 1e-3)
            self.terms["charbonnier"] = CharbonnierLoss(eps=eps)
            self.weights["charbonnier"] = w

        # Edge (Sobel)
        edge_cfg = loss_cfg.get("edge", {})
        w = edge_cfg.get("weight", 0.1)
        if w > 0:
            self.terms["edge"] = EdgeLoss()
            self.weights["edge"] = w

        # FFT Magnitude (anti-hallucination)
        fft_cfg = loss_cfg.get("fft", {})
        w = fft_cfg.get("weight", 0.05)
        if w > 0:
            self.terms["fft"] = FFTLoss()
            self.weights["fft"] = w

        # Range Penalty
        rp_cfg = loss_cfg.get("range_penalty", {})
        w = rp_cfg.get("weight", 0.01)
        if w > 0:
            self.terms["range_penalty"] = RangePenaltyLoss()
            self.weights["range_penalty"] = w

        # Total Variation
        tv_cfg = loss_cfg.get("tv", {})
        w = tv_cfg.get("weight", 0.001)
        if w > 0:
            self.terms["tv"] = TVLoss()
            self.weights["tv"] = w

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> tuple:
        """
        Compute weighted composite loss.

        Args:
            pred: Model output, shape (B, 1, H, W).
            gt: Ground truth, shape (B, 1, H, W).

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains each
            individual term's unweighted value for logging/dashboard.
        """
        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        loss_dict = {}

        for name, module in self.terms.items():
            term_val = module(pred, gt)
            loss_dict[name] = term_val.item()
            total = total + self.weights[name] * term_val

        loss_dict["total"] = total.item()
        return total, loss_dict

    def __repr__(self) -> str:
        lines = ["CompositeLoss("]
        for name in self.terms:
            lines.append(f"  {name}: weight={self.weights[name]}")
        lines.append(")")
        return "\n".join(lines)
