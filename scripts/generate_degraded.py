"""
Generate synthetic degraded images from clean GT images.

Usage:
    python scripts/generate_degraded.py \
        --gt_dir data/train/gt \
        --output_dir data/train/degraded \
        --config configs/hansr.yaml \
        [--case all_three]

Generates one degraded image per GT image, preserving filenames.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from hansr.dataset import discover_images, load_grayscale
from hansr.degradation import degrade_image, DEGRADATION_CASES
from hansr.utils import load_config, set_seed


def save_grayscale(tensor: torch.Tensor, path: str) -> None:
    """Save a single-channel tensor as a grayscale image."""
    arr = tensor.squeeze().numpy()
    # Clip for saving (display only) — network input is NOT clipped (FR-004)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def main():
    parser = argparse.ArgumentParser(description="Generate degraded images from GT")
    parser.add_argument("--gt_dir", required=True, help="Ground truth image directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for degraded images")
    parser.add_argument("--config", default="configs/hansr.yaml", help="Config file")
    parser.add_argument("--case", default=None, help="Specific degradation case (e.g., 'speckle', 'gaussian', 'all_three')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    deg_cfg = config["degradation"]
    set_seed(args.seed)

    # Parse case
    case_map = {
        "speckle": ("speckle",),
        "gaussian": ("gaussian",),
        "resolution": ("resolution",),
        "speckle_gaussian": ("speckle", "gaussian"),
        "speckle_resolution": ("speckle", "resolution"),
        "gaussian_resolution": ("gaussian", "resolution"),
        "all_three": ("speckle", "gaussian", "resolution"),
    }
    case = case_map.get(args.case) if args.case else None

    gt_files = discover_images(args.gt_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating degraded images: {len(gt_files)} files")
    print(f"  Case: {case if case else 'random (all 7)'}")

    for gt_path in tqdm(gt_files, desc="Degrading"):
        gt = load_grayscale(gt_path)
        degraded, params = degrade_image(gt, deg_cfg, case=case)

        filename = os.path.basename(gt_path)
        out_path = os.path.join(args.output_dir, filename)
        save_grayscale(degraded, out_path)

    print(f"Done. {len(gt_files)} degraded images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
