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


def discover_images(directory: Optional[str]) -> List[str]:
    """Discover all supported image files in a directory, sorted for determinism."""
    if not directory:
        return []
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        return []

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
        elif arr.ndim == 3 and arr.shape[2] == 1:
            tensor = torch.from_numpy(arr.squeeze(2)).unsqueeze(0)  # (1, H, W)
        else:
            raise ValueError(
                f"Unexpected npy array shape {arr.shape} in {path}, expected (H, W), (1, H, W) or (H, W, 1)"
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
# PairedDataset — Pre-paired GT/Degraded from Disk or File Lists
# =============================================================================

class PairedDataset(Dataset):
    """
    Loads pre-paired GT and degraded images from two directories or file lists.

    Pairing is by sorted filename order — filenames must match between
    gt and degraded.

    Args:
        gt_dir: Path to ground truth images (optional if gt_files provided).
        degraded_dir: Path to degraded images (optional if degraded_files provided).
        crop_size: Random crop size (from degraded). None = no crop (full image).
        augment: Apply random flips.
        scale: Resolution ratio GT/degraded (default 2).
        gt_files: Explicit list of GT image file paths.
        degraded_files: Explicit list of degraded image file paths.
    """

    def __init__(
        self,
        gt_dir: Optional[str] = None,
        degraded_dir: Optional[str] = None,
        crop_size: Optional[int] = None,
        augment: bool = True,
        scale: int = 2,
        gt_files: Optional[List[str]] = None,
        degraded_files: Optional[List[str]] = None,
    ):
        if gt_files is not None and degraded_files is not None:
            self.gt_files = list(gt_files)
            self.degraded_files = list(degraded_files)
        else:
            self.gt_files = discover_images(gt_dir)
            self.degraded_files = discover_images(degraded_dir)

        self.crop_size = crop_size
        self.augment = augment
        self.scale = scale

        if len(self.gt_files) == 0:
            raise ValueError(f"No supported images found in GT: {gt_dir or gt_files}")
        if len(self.degraded_files) == 0:
            raise ValueError(f"No supported images found in degraded: {degraded_dir or degraded_files}")

        if len(self.gt_files) != len(self.degraded_files):
            raise ValueError(
                f"GT ({len(self.gt_files)}) and degraded ({len(self.degraded_files)}) "
                f"image counts don't match"
            )

        logger.info(
            f"PairedDataset: {len(self.gt_files)} pairs loaded "
            f"(crop_size={self.crop_size}, augment={self.augment})"
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

def build_datasets(config: dict) -> Dict[str, Optional[Dataset]]:
    """
    Build train/val datasets from config (FR-008).

    If config.data.use_synthetic_degradation is True, uses SyntheticDataset
    (GT-only, degrade on-the-fly). Otherwise uses PairedDataset (pre-paired).

    When val_gt_dir is not provided or empty on disk, automatically generates
    a deterministic, leakage-free 90/10 train/validation split (seed 42)
    from the discovered paired training data.

    Returns:
        Dict with 'train' (Dataset) and optionally 'val' (Dataset or None).
    """
    import random
    data_cfg = config["data"]
    train_cfg = config["training"]
    deg_cfg = config["degradation"]
    crop_size = train_cfg.get("patch_size", 128)
    scale = config["model"].get("upscale_factor", 2)
    use_synthetic = data_cfg.get("use_synthetic_degradation", False)
    seed = train_cfg.get("seed", 42)

    train_gt = data_cfg.get("train_gt_dir", "data/train/gt")
    train_deg = data_cfg.get("train_degraded_dir", "data/train/degraded")

    # Auto-resolve Kaggle KLA dataset paths if local defaults are absent
    kaggle_gt = "/kaggle/input/datasets/jhansiranimajhi/kla-dataset/train/GT"
    kaggle_deg = "/kaggle/input/datasets/jhansiranimajhi/kla-dataset/train/NoisyLR"
    if not os.path.exists(train_gt) and os.path.exists(kaggle_gt):
        train_gt = kaggle_gt
        train_deg = kaggle_deg
        logger.info(f"Auto-resolved Kaggle KLA paths: {train_gt} / {train_deg}")

    datasets: Dict[str, Optional[Dataset]] = {}

    # Check if explicit validation directories exist and have matching images
    val_gt = data_cfg.get("val_gt_dir")
    val_deg = data_cfg.get("val_degraded_dir")
    val_gt_files = discover_images(val_gt) if val_gt and os.path.exists(val_gt) else []
    val_deg_files = discover_images(val_deg) if val_deg and os.path.exists(val_deg) else []

    has_explicit_val = (
        (use_synthetic and len(val_gt_files) > 0)
        or (not use_synthetic and len(val_gt_files) > 0 and len(val_gt_files) == len(val_deg_files))
    )

    if use_synthetic:
        datasets["train"] = SyntheticDataset(
            gt_dir=train_gt,
            degradation_config=deg_cfg,
            crop_size=crop_size,
            augment=True,
            scale=scale,
        )
        if has_explicit_val:
            datasets["val"] = SyntheticDataset(
                gt_dir=val_gt,
                degradation_config=deg_cfg,
                crop_size=None,  # full images for validation
                augment=False,
                scale=scale,
            )
        else:
            datasets["val"] = None
    else:
        # Paired dataset mode
        if has_explicit_val:
            datasets["train"] = PairedDataset(
                gt_dir=train_gt,
                degraded_dir=train_deg,
                crop_size=crop_size,
                augment=True,
                scale=scale,
            )
            datasets["val"] = PairedDataset(
                gt_dir=val_gt,
                degraded_dir=val_deg,
                crop_size=None,
                augment=False,
                scale=scale,
            )
        else:
            # Deterministic, leakage-free 90/10 split over discovered paired images
            all_gt_files = discover_images(train_gt)
            all_deg_files = discover_images(train_deg)

            if len(all_gt_files) == 0:
                raise ValueError(f"No supported images found in GT directory: {train_gt}")
            if len(all_deg_files) == 0:
                raise ValueError(f"No supported images found in degraded directory: {train_deg}")
            if len(all_gt_files) != len(all_deg_files):
                raise ValueError(
                    f"GT ({len(all_gt_files)}) and degraded ({len(all_deg_files)}) counts don't match"
                )

            # Match files strictly by stem to ensure GT and degraded remain paired
            deg_map = {Path(f).stem: f for f in all_deg_files}
            paired_gt = []
            paired_deg = []
            for gf in all_gt_files:
                stem = Path(gf).stem
                if stem in deg_map:
                    paired_gt.append(gf)
                    paired_deg.append(deg_map[stem])
                else:
                    raise ValueError(f"Missing degraded match for GT file: {gf}")

            val_ratio = data_cfg.get("val_ratio", 0.10)
            if val_ratio > 0.0 and len(paired_gt) >= 2:
                # Deterministic split using seed 42
                rng = random.Random(seed)
                n = len(paired_gt)
                indices = list(range(n))
                rng.shuffle(indices)

                n_val = int(round(n * val_ratio))
                n_train = n - n_val
                val_idx_set = set(indices[:n_val])

                train_gt_list = [paired_gt[i] for i in range(n) if i not in val_idx_set]
                train_deg_list = [paired_deg[i] for i in range(n) if i not in val_idx_set]
                val_gt_list = [paired_gt[i] for i in range(n) if i in val_idx_set]
                val_deg_list = [paired_deg[i] for i in range(n) if i in val_idx_set]

                # Leakage check
                assert len(set(train_gt_list) & set(val_gt_list)) == 0, "Train/val leakage detected!"
                assert len(train_gt_list) == n_train, f"Expected {n_train} train pairs, got {len(train_gt_list)}"
                assert len(val_gt_list) == n_val, f"Expected {n_val} val pairs, got {len(val_gt_list)}"

                logger.info(
                    f"Generated leakage-free {int((1-val_ratio)*100)}/{int(val_ratio*100)} split: "
                    f"Train={len(train_gt_list)} pairs, Val={len(val_gt_list)} pairs (seed={seed})"
                )

                datasets["train"] = PairedDataset(
                    crop_size=crop_size,
                    augment=True,
                    scale=scale,
                    gt_files=train_gt_list,
                    degraded_files=train_deg_list,
                )
                datasets["val"] = PairedDataset(
                    crop_size=None,
                    augment=False,
                    scale=scale,
                    gt_files=val_gt_list,
                    degraded_files=val_deg_list,
                )
            else:
                datasets["train"] = PairedDataset(
                    gt_dir=train_gt,
                    degraded_dir=train_deg,
                    crop_size=crop_size,
                    augment=True,
                    scale=scale,
                )
                datasets["val"] = None

    return datasets
