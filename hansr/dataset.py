"""
HANSR Dataset — Paired Image Loading and Synthetic Degradation (FR-001, FR-003, FR-004)

Two dataset modes:
  1. PairedDataset — loads pre-paired GT/degraded images from disk
  2. SyntheticDataset — takes clean GT images and degrades on-the-fly

Both enforce:
  - Single-channel grayscale throughout (FR-003)
  - No clipping of degraded values (FR-004)
  - Random cropping and augmentation for training
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from hansr.degradation import degrade_image, DEGRADATION_CASES

logger = logging.getLogger("hansr")


# =============================================================================
# Image I/O Utilities
# =============================================================================

SUPPORTED_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".npy"}


def discover_images(directory: str) -> List[str]:
    """Discover all supported image files in a directory, sorted for determinism."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    files = []
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file():
            files.append(str(f))

    return files


def load_grayscale(path: str) -> torch.Tensor:
    """
    Load an image as a single-channel float32 tensor.

    For standard image files (.png, .jpg, .tif, etc.), converts to grayscale
    and normalizes to [0, 1].
    For .npy files, loads raw array as float32 with shape (1, H, W) without
    additional normalization or clipping.

    Args:
        path: Path to image file.

    Returns:
        Tensor of shape (1, H, W), float32.

    Raises:
        ValueError: If image cannot be loaded or decoded.
    """
    if str(path).lower().endswith(".npy"):
        try:
            arr = np.load(path)
        except Exception as e:
            raise ValueError(f"Cannot open npy file {path}: {e}")

        arr = arr.astype(np.float32)

        if arr.ndim == 2:
            tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            tensor = torch.from_numpy(arr)
        else:
            raise ValueError(
                f"Unexpected npy array shape {arr.shape} in {path}, expected (H, W) or (1, H, W)"
            )

        return tensor

    try:
        img = Image.open(path)
    except Exception as e:
        raise ValueError(f"Cannot open image {path}: {e}")

    # Convert to grayscale (FR-003)
    img = img.convert("L")

    # To float32 tensor, normalized to [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

    return tensor


# =============================================================================
# Augmentation
# =============================================================================

def random_crop_pair(
    gt: torch.Tensor,
    degraded: torch.Tensor,
    crop_size: int,
    scale: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Random crop from paired GT/degraded images with correct spatial alignment.

    GT is at 2× the resolution of degraded, so we crop at (crop_size) from
    degraded and (crop_size * scale) from GT at the corresponding location.

    Args:
        gt: Ground truth tensor (1, H_gt, W_gt).
        degraded: Degraded tensor (1, H_deg, W_deg) where H_deg = H_gt/scale.
        crop_size: Crop size for the degraded image.
        scale: Resolution ratio GT/degraded.

    Returns:
        (gt_crop, degraded_crop) properly aligned.
    """
    _, h_deg, w_deg = degraded.shape
    gt_crop_size = crop_size * scale

    # Random top-left corner in degraded space
    top = torch.randint(0, max(1, h_deg - crop_size + 1), (1,)).item()
    left = torch.randint(0, max(1, w_deg - crop_size + 1), (1,)).item()

    # Crop degraded
    degraded_crop = degraded[:, top:top + crop_size, left:left + crop_size]

    # Corresponding crop in GT space
    gt_top = top * scale
    gt_left = left * scale
    gt_crop = gt[:, gt_top:gt_top + gt_crop_size, gt_left:gt_left + gt_crop_size]

    return gt_crop, degraded_crop


def random_augment(
    gt: torch.Tensor, degraded: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random horizontal/vertical flips (consistent for both)."""
    if torch.rand(1).item() > 0.5:
        gt = torch.flip(gt, [2])       # horizontal flip
        degraded = torch.flip(degraded, [2])
    if torch.rand(1).item() > 0.5:
        gt = torch.flip(gt, [1])       # vertical flip
        degraded = torch.flip(degraded, [1])
    return gt, degraded


# =============================================================================
# PairedDataset — Pre-paired GT/Degraded from Disk
# =============================================================================

class PairedDataset(Dataset):
    """
    Loads pre-paired GT and degraded images from two directories.

    Pairing is by sorted filename order — filenames must match between
    gt_dir and degraded_dir.

    Args:
        gt_dir: Path to ground truth images.
        degraded_dir: Path to degraded images.
        crop_size: Random crop size (from degraded). None = no crop (full image).
        augment: Apply random flips.
        scale: Resolution ratio GT/degraded (default 2).
    """

    def __init__(
        self,
        gt_dir: str,
        degraded_dir: str,
        crop_size: Optional[int] = None,
        augment: bool = True,
        scale: int = 2,
    ):
        self.gt_files = discover_images(gt_dir)
        self.degraded_files = discover_images(degraded_dir)
        self.crop_size = crop_size
        self.augment = augment
        self.scale = scale

        if len(self.gt_files) != len(self.degraded_files):
            raise ValueError(
                f"GT ({len(self.gt_files)}) and degraded ({len(self.degraded_files)}) "
                f"image counts don't match"
            )

        if len(self.gt_files) == 0:
            raise ValueError(f"No images found in {gt_dir} / {degraded_dir}")

        logger.info(
            f"PairedDataset: {len(self.gt_files)} pairs from "
            f"{gt_dir} / {degraded_dir}"
        )

    def __len__(self) -> int:
        return len(self.gt_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gt = load_grayscale(self.gt_files[idx])
        degraded = load_grayscale(self.degraded_files[idx])

        if self.crop_size is not None:
            gt, degraded = random_crop_pair(gt, degraded, self.crop_size, self.scale)

        if self.augment:
            gt, degraded = random_augment(gt, degraded)

        return {
            "gt": gt,
            "degraded": degraded,
            "gt_path": self.gt_files[idx],
            "degraded_path": self.degraded_files[idx],
        }


# =============================================================================
# SyntheticDataset — On-the-Fly Degradation from Clean GT Only
# =============================================================================

class SyntheticDataset(Dataset):
    """
    Loads clean GT images and generates degraded versions on-the-fly.

    Uses the full 7-case degradation pipeline. Every epoch sees different
    random degradation parameters, providing implicit data augmentation.

    The GT images should be at the target (high) resolution. The degradation
    pipeline will downsample as needed.

    Args:
        gt_dir: Path to clean ground truth images (high resolution).
        degradation_config: Degradation section of YAML config.
        crop_size: Random crop size (from the GT image, pre-degradation).
            The actual degraded crop will be crop_size/scale.
        augment: Apply random flips.
        scale: Downsampling factor (default 2).
    """

    def __init__(
        self,
        gt_dir: str,
        degradation_config: dict,
        crop_size: Optional[int] = None,
        augment: bool = True,
        scale: int = 2,
    ):
        self.gt_files = discover_images(gt_dir)
        self.degradation_config = degradation_config
        self.crop_size = crop_size
        self.augment = augment
        self.scale = scale

        if len(self.gt_files) == 0:
            raise ValueError(f"No images found in {gt_dir}")

        logger.info(
            f"SyntheticDataset: {len(self.gt_files)} GT images from {gt_dir}, "
            f"degradation on-the-fly"
        )

    def __len__(self) -> int:
        return len(self.gt_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gt = load_grayscale(self.gt_files[idx])

        # Crop from GT first (at high resolution), then degrade
        if self.crop_size is not None:
            gt_crop_size = self.crop_size * self.scale  # crop in GT space
            _, h, w = gt.shape
            if h >= gt_crop_size and w >= gt_crop_size:
                top = torch.randint(0, max(1, h - gt_crop_size + 1), (1,)).item()
                left = torch.randint(0, max(1, w - gt_crop_size + 1), (1,)).item()
                gt = gt[:, top:top + gt_crop_size, left:left + gt_crop_size]

        # Apply random degradation (all 7 cases, random severity)
        degraded, params = degrade_image(gt, self.degradation_config)

        if self.augment:
            gt, degraded = random_augment(gt, degraded)

        return {
            "gt": gt,
            "degraded": degraded,
            "gt_path": self.gt_files[idx],
            "degradation_case": str(params["case"]),
        }


# =============================================================================
# Dataset Factory
# =============================================================================

def build_datasets(config: dict) -> Dict[str, Dataset]:
    """
    Build train/val datasets from config (FR-008).

    If config.data.use_synthetic_degradation is True, uses SyntheticDataset
    (GT-only, degrade on-the-fly). Otherwise uses PairedDataset (pre-paired).

    Returns:
        Dict with 'train' and 'val' Dataset instances.
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    deg_cfg = config["degradation"]
    crop_size = train_cfg.get("patch_size", 128)
    scale = config["model"].get("upscale_factor", 2)
    use_synthetic = data_cfg.get("use_synthetic_degradation", False)

    datasets = {}

    if use_synthetic:
        datasets["train"] = SyntheticDataset(
            gt_dir=data_cfg["train_gt_dir"],
            degradation_config=deg_cfg,
            crop_size=crop_size,
            augment=True,
            scale=scale,
        )
        datasets["val"] = SyntheticDataset(
            gt_dir=data_cfg["val_gt_dir"],
            degradation_config=deg_cfg,
            crop_size=None,  # full images for validation
            augment=False,
            scale=scale,
        )
    else:
        datasets["train"] = PairedDataset(
            gt_dir=data_cfg["train_gt_dir"],
            degraded_dir=data_cfg["train_degraded_dir"],
            crop_size=crop_size,
            augment=True,
            scale=scale,
        )
        datasets["val"] = PairedDataset(
            gt_dir=data_cfg["val_gt_dir"],
            degraded_dir=data_cfg["val_degraded_dir"],
            crop_size=None,
            augment=False,
            scale=scale,
        )

    return datasets
