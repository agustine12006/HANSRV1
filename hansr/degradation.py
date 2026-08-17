"""
HANSR Degradation Pipeline (FR-001, FR-004)

Implements the three degradation mechanisms specified in the PRD:
  1. Additive Gaussian noise:  Y = X + eps,  eps ~ N(0, sigma_g^2)
  2. Multiplicative speckle:   Y = X + X*N,  N ~ N(0, sigma_s^2)
  3. Bicubic 2x downsample

Applied in all 7 combinations (blind, random during training).

CRITICAL (FR-004): Degraded pixel values may exceed [0,1] — this is intentional
and must NOT be clipped before the network sees the input.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# Individual Degradation Operations
# =============================================================================

def add_gaussian_noise(
    image: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Additive Gaussian noise: Y = X + eps, eps ~ N(0, sigma^2).
    Applied after downsampling per PRD spec [Ch.38.2].
    """
    noise = torch.randn_like(image) * sigma
    return image + noise


def add_speckle_noise(
    image: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Multiplicative speckle noise: Y = X + X * N, N ~ N(0, sigma^2).
    Models coherent-imaging artifact where noise intensity scales with signal.
    Output may exceed [0,1] — this is correct behavior (FR-004).
    """
    noise = torch.randn_like(image) * sigma
    return image + image * noise


def bicubic_downsample(
    image: torch.Tensor,
    factor: int = 2,
) -> torch.Tensor:
    """
    Bicubic 2x downsampling. Reduces spatial resolution by factor.
    Input: (B, C, H, W) or (C, H, W) -> output at H/factor, W/factor.
    """
    needs_batch = image.ndim == 3
    if needs_batch:
        image = image.unsqueeze(0)

    downsampled = F.interpolate(
        image,
        scale_factor=1.0 / factor,
        mode="bicubic",
        align_corners=False,
    )

    if needs_batch:
        downsampled = downsampled.squeeze(0)
    return downsampled


# =============================================================================
# Seven-Case Degradation Combinations
# =============================================================================

# All 7 possible non-empty combinations of {speckle, gaussian, resolution}
DEGRADATION_CASES = [
    ("speckle",),                          # case 1: speckle only
    ("gaussian",),                         # case 2: gaussian only
    ("resolution",),                       # case 3: resolution only
    ("speckle", "gaussian"),               # case 4: speckle + gaussian
    ("speckle", "resolution"),             # case 5: speckle + resolution
    ("gaussian", "resolution"),            # case 6: gaussian + resolution
    ("speckle", "gaussian", "resolution"), # case 7: all three
]


def sample_degradation_params(
    config: dict,
    case: Optional[Tuple[str, ...]] = None,
) -> Dict:
    """
    Sample random degradation parameters for one image.

    Args:
        config: degradation section of YAML config.
        case: Specific degradation case to apply. If None, randomly selects
              from all 7 cases (blind multi-degradation, FR-001).

    Returns:
        Dict with 'case', 'gaussian_sigma', 'speckle_sigma', 'downsample_factor'.
    """
    if case is None:
        case = random.choice(DEGRADATION_CASES)

    params = {
        "case": case,
        "gaussian_sigma": 0.0,
        "speckle_sigma": 0.0,
        "downsample_factor": 1,
    }

    if "gaussian" in case:
        g_range = config.get("gaussian", {}).get("sigma_range", [0.01, 0.08])
        params["gaussian_sigma"] = random.uniform(g_range[0], g_range[1])

    if "speckle" in case:
        s_range = config.get("speckle", {}).get("sigma_range", [0.05, 0.3])
        params["speckle_sigma"] = random.uniform(s_range[0], s_range[1])

    if "resolution" in case:
        params["downsample_factor"] = config.get("downsample", {}).get("factor", 2)

    return params


# =============================================================================
# Apply Degradation Pipeline
# =============================================================================

def apply_degradation(
    clean_image: torch.Tensor,
    params: Dict,
) -> torch.Tensor:
    """
    Apply a specific degradation combination to a clean image.

    Order of operations (per PRD [Ch.38.2]):
      1. Downsample (if resolution degradation)
      2. Speckle noise (multiplicative, applied to signal)
      3. Gaussian noise (additive, applied last)

    Args:
        clean_image: Clean GT image tensor, shape (C, H, W) or (B, C, H, W).
            Values in [0, 1].
        params: Degradation parameters from sample_degradation_params().

    Returns:
        Degraded image. Shape may differ from input if downsampled.
        Values may exceed [0, 1] — this is correct (FR-004).
    """
    degraded = clean_image.clone()

    # Step 1: Downsample
    if params["downsample_factor"] > 1:
        degraded = bicubic_downsample(degraded, params["downsample_factor"])

    # Step 2: Speckle noise (multiplicative)
    if params["speckle_sigma"] > 0:
        degraded = add_speckle_noise(degraded, params["speckle_sigma"])

    # Step 3: Gaussian noise (additive)
    if params["gaussian_sigma"] > 0:
        degraded = add_gaussian_noise(degraded, params["gaussian_sigma"])

    return degraded


def degrade_image(
    clean_image: torch.Tensor,
    config: dict,
    case: Optional[Tuple[str, ...]] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    High-level API: sample random params and degrade a clean image.

    Args:
        clean_image: Clean GT tensor.
        config: Degradation config section.
        case: Optional specific case; if None, random.

    Returns:
        (degraded_image, params_dict)
    """
    params = sample_degradation_params(config, case=case)
    degraded = apply_degradation(clean_image, params)
    return degraded, params
