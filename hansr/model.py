"""
HANSR Model Architecture — NAFNet-style Backbone for Semiconductor Image Restoration

Architecture: encoder-decoder UNet with NAFBlocks (SimpleGate + SCA),
terminal PixelShuffle 2x reconstruction head, residual-around-fixed-bicubic design.

Requirements satisfied:
  FR-001: Blind multi-degradation restoration (single forward pass)
  FR-002: Input/Output resolution contract (128->256, 256->512)
  FR-003: Grayscale-only pipeline (1-channel in/out)
  FR-005: Residual learning with fixed bicubic anchor (no sigmoid)
  FR-006: Anti-hallucination (residual design, no invented texture)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# =============================================================================
# Building Blocks
# =============================================================================

class LayerNorm2d(nn.Module):
    """Channel-wise Layer Normalization for 2D feature maps."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize over channel dimension per spatial location
        mu = x.mean(1, keepdim=True)
        sigma = x.var(1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    """
    Nonlinear activation replacement from NAFNet.
    Splits channels in half and multiplies element-wise — creates nonlinearity
    without explicit activation functions (ReLU/GELU/Sigmoid).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """
    SCA from NAFNet — lightweight channel attention.
    Global average pool -> 1x1 conv -> element-wise multiply.
    No MLP, no sigmoid, no complex gating.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


# =============================================================================
# NAFBlock — Core Building Block
# =============================================================================

class NAFBlock(nn.Module):
    """
    NAFNet Block: the fundamental processing unit.

    Structure:
      Path 1 (spatial): LN -> 1x1 Conv(C->2C) -> 3x3 DWConv(2C) -> LHF Enhancement
                         -> SimpleGate(->C) -> SCA(C) -> 1x1 Conv(C) -> dropout -> residual(* beta)
      Path 2 (channel): LN -> 1x1 Conv(C->2C) -> SimpleGate(->C)
                         -> 1x1 Conv(C) -> dropout -> residual(* gamma)

    Beta/gamma are learnable per-channel scaling for residual connections.
    """

    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2,
                 dropout_rate: float = 0.0):
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        # --- Spatial mixing path ---
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1, 1, 0)       # pointwise expand
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, 1, 1,
                               groups=dw_channels)                     # depthwise spatial
        # Local High-Frequency detail scaling (learnable per-channel)
        self.hf_scale = nn.Parameter(torch.zeros(1, dw_channels, 1, 1))
        self.sg1 = SimpleGate()                                        # 2C -> C
        self.sca = SimplifiedChannelAttention(dw_channels // 2)        # attention on C
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1, 1, 0)   # pointwise project

        # --- Channel mixing path (FFN) ---
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1, 1, 0)       # pointwise expand
        self.sg2 = SimpleGate()                                        # 2C -> C
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1, 1, 0)  # pointwise project

        # --- Learnable residual scaling ---
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # --- Dropout ---
        self.drop1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.drop2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # Spatial mixing with local high-frequency enhancement
        x = self.norm1(inp)
        x = self.conv1(x)
        x_dw = self.conv2(x)
        # Laplacian-style local high-pass feature difference
        x_hf = x_dw - F.avg_pool2d(x_dw, kernel_size=3, stride=1, padding=1)
        x = x_dw + self.hf_scale * x_hf
        x = self.sg1(x)
        x = self.sca(x)
        x = self.conv3(x)
        x = self.drop1(x)
        y = inp + x * self.beta

        # Channel mixing (FFN)
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.drop2(x)
        return y + x * self.gamma


class DetailSkipFusion(nn.Module):
    """
    Detail-Preserving Skip Connection Fusion.

    Fuses concatenated decoder and encoder skip features while preserving
    local high-frequency edge transitions via lightweight depthwise gating.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)
        self.dw_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Sigmoid(),
        )

    def forward(self, cat_feat: torch.Tensor) -> torch.Tensor:
        fused = self.proj(cat_feat)
        skip = cat_feat[:, fused.shape[1] :, :, :]
        gate = self.dw_gate(fused)
        return fused + skip * gate


class LightweightReconstructionHead(nn.Module):
    """
    Lightweight Sub-Pixel Reconstruction Head.

    Provides pre-PixelShuffle spatial feature refinement via depthwise separable
    convolution with residual shortcut, enabling sharper sub-pixel detail
    projection without introducing heavy global attention or large channel expansion.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 1,
        upscale_factor: int = 2,
    ):
        super().__init__()
        ps_channels = out_channels * (upscale_factor ** 2)

        # Lightweight pre-shuffle feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
            ),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Terminal sub-pixel projection
        self.proj = nn.Conv2d(in_channels, ps_channels, kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = x + self.refine(x)
        return self.ps(self.proj(feat))


# =============================================================================
# HANSRNet — Full Model
# =============================================================================

class HANSRNet(nn.Module):
    """
    HANSR: NAFNet-style encoder-decoder for semiconductor image restoration.

    Architecture:
      Input (1ch, H x W)
        |-- Fixed bicubic upsample (frozen) -----> (1ch, 2H x 2W)
        |-- Learnable branch:                              |
        |     Stem: 3x3 Conv(1 -> C)                       |
        |     Encoder: [N x LHF-NAFBlock + downsample] x L |
        |     Middle: M x LHF-NAFBlock                     |
        |     Decoder: [upsample + DetailSkip + NAFBlock]  |
        |     Reconstruction: LightweightReconstructionHead|
        |                              |                   |
        |                         residual (1ch, 2H x 2W)  |
        |                              |                   |
        Output = bicubic_anchor + residual  (no sigmoid, no clamp)

    Args:
        in_channels: Input channels (1 for grayscale). FR-003.
        out_channels: Output channels (1 for grayscale). FR-003.
        width: Base channel width for NAFBlocks.
        num_blocks: List of block counts per encoder stage.
        middle_blocks: Number of NAFBlocks in the bottleneck.
        dropout_rate: Dropout rate inside NAFBlocks (0 = off).
        upscale_factor: Super-resolution factor (2). FR-002.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        num_blocks: List[int] = None,
        middle_blocks: int = 4,
        dropout_rate: float = 0.0,
        upscale_factor: int = 2,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 4, 4, 8]

        self.upscale_factor = upscale_factor
        num_stages = len(num_blocks)

        # --- Stem ---
        self.intro = nn.Conv2d(in_channels, width, 3, 1, 1)

        # --- Encoder ---
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for i in range(num_stages):
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan, dropout_rate=dropout_rate)
                                for _ in range(num_blocks[i])])
            )
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan *= 2

        # --- Middle / Bottleneck ---
        self.middle = nn.Sequential(
            *[NAFBlock(chan, dropout_rate=dropout_rate) for _ in range(middle_blocks)]
        )

        # --- Decoder ---
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.skip_projs = nn.ModuleList()  # Detail-preserving skip connections
        dec_block_nums = list(reversed(num_blocks))
        for i in range(num_stages):
            # Upsample: chan -> chan//2 at 2x spatial via PixelShuffle
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, 1, 0),  # expand channels for PS
                    nn.PixelShuffle(2),                  # chan*2 -> chan*2/4 = chan//2
                )
            )
            chan //= 2
            # Detail-preserving skip fusion
            self.skip_projs.append(DetailSkipFusion(chan))
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan, dropout_rate=dropout_rate)
                                for _ in range(dec_block_nums[i])])
            )

        # --- Terminal Reconstruction Head (FR-005) ---
        # Lightweight pre-refinement + PixelShuffle for 2x spatial upsample
        self.reconstruction = LightweightReconstructionHead(
            in_channels=chan,
            out_channels=out_channels,
            upscale_factor=upscale_factor,
        )

        # No sigmoid — residual is unbounded (FR-005, FR-006)

    def _bicubic_upsample(self, x: torch.Tensor) -> torch.Tensor:
        """Fixed, non-learnable bicubic upsample anchor (FR-005)."""
        return F.interpolate(
            x,
            scale_factor=self.upscale_factor,
            mode="bicubic",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: single model, single pass, degradation-blind (FR-001).

        Args:
            x: Degraded grayscale input, shape (B, 1, H, W).

        Returns:
            Restored image, shape (B, 1, 2H, 2W). No sigmoid, no clamp.
        """
        # Fixed bicubic anchor (frozen, non-trainable)
        bicubic = self._bicubic_upsample(x)

        # Learnable restoration branch
        feat = self.intro(x)

        # Encoder — collect skip connections
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            feat = encoder(feat)
            skips.append(feat)
            feat = down(feat)

        # Bottleneck
        feat = self.middle(feat)

        # Decoder — fuse skip connections
        skips.reverse()
        for up, skip_proj, decoder, skip in zip(
            self.ups, self.skip_projs, self.decoders, skips
        ):
            feat = up(feat)
            feat = skip_proj(torch.cat([feat, skip], dim=1))
            feat = decoder(feat)

        # Terminal reconstruction head: residual at 2x resolution
        residual = self.reconstruction(feat)

        # Output = fixed bicubic + learned residual (FR-005)
        return bicubic + residual


# =============================================================================
# Factory Function
# =============================================================================

def build_model(config: dict) -> HANSRNet:
    """
    Build HANSRNet from a config dictionary (FR-008).

    Args:
        config: Parsed YAML config with 'model' section.

    Returns:
        Initialized HANSRNet instance.
    """
    model_cfg = config["model"]
    model = HANSRNet(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        width=model_cfg.get("width", 32),
        num_blocks=model_cfg.get("num_blocks", [2, 4, 4, 8]),
        middle_blocks=model_cfg.get("middle_blocks", 4),
        dropout_rate=model_cfg.get("dropout_rate", 0.0),
        upscale_factor=model_cfg.get("upscale_factor", 2),
    )
    return model


# =============================================================================
# Model Info Utility
# =============================================================================

def count_parameters(model: nn.Module) -> dict:
    """Count total, trainable, and non-trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "total_mb": total * 4 / (1024 ** 2),  # float32
    }
