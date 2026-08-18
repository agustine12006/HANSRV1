"""
Leakage-Free Split Generation (FR-010)

Groups all patches from the same source/specimen image into one split.
Generates train/val/test/OOD assignments and outputs a splits.json manifest.
Supports standard image formats (.png, .tif, .tiff, .jpg, .jpeg, .bmp) and raw array datasets (.npy).

Usage:
    python scripts/generate_splits.py \
        --image_dir data/all_gt \
        --output splits.json \
        --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
from hansr.utils import set_seed, ensure_dir


def extract_source_id(filename: str) -> str:
    """
    Extract the source/specimen identity from a filename.

    Strategy: strip any patch/crop suffix to find the parent image ID.
    Handles patterns like 'specimen_001_patch_003.png' -> 'specimen_001'
    or 'wafer_01_crop_02.npy' -> 'wafer_01'.
    Falls back to the full stem if no pattern is detected.
    """
    stem = Path(filename).stem

    # Try common patterns: *_patch_NNN, *_crop_NNN, *_pNNN, *_NNN (at end)
    for pattern in [r"(.+?)_patch_\d+", r"(.+?)_crop_\d+", r"(.+?)_p\d+$"]:
        m = re.match(pattern, stem)
        if m:
            return m.group(1)

    return stem


def generate_splits(
    image_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Generate leakage-free train/val/test splits.

    All files from the same source identity go into the same split.
    Supports standard image formats (.png, .tif, .tiff, .jpg, .jpeg, .bmp)
    and raw array datasets (.npy). Each .npy file is treated as one complete
    source/specimen file without decoding, slicing, or converting array contents.

    Returns:
        Dict with 'train', 'val', 'test' lists of filenames,
        plus metadata about the split.
    """
    set_seed(seed)
    image_path = Path(image_dir)
    if not image_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".npy"}
    files = sorted([f.name for f in image_path.iterdir()
                    if f.suffix.lower() in exts and f.is_file()])

    if len(files) == 0:
        raise ValueError(f"No supported images found in {image_dir} (supported extensions: {exts})")

    # Group by source identity
    groups = defaultdict(list)
    for f in files:
        source_id = extract_source_id(f)
        groups[source_id].append(f)

    # Shuffle source IDs and split
    source_ids = sorted(groups.keys())
    random.shuffle(source_ids)

    n = len(source_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = source_ids[:n_train]
    val_ids = source_ids[n_train:n_train + n_val]
    test_ids = source_ids[n_train + n_val:]

    train_files = [f for sid in train_ids for f in groups[sid]]
    val_files = [f for sid in val_ids for f in groups[sid]]
    test_files = [f for sid in test_ids for f in groups[sid]]

    # Completeness check: all files are assigned
    all_assigned = train_files + val_files + test_files
    assert len(all_assigned) == len(files), f"Total files mismatch: {len(all_assigned)} assigned vs {len(files)} discovered"
    assert set(all_assigned) == set(files), "Mismatch between discovered files and assigned split files"

    # Leakage check
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)
    assert len(train_set & val_set) == 0, "Train/val leakage detected!"
    assert len(train_set & test_set) == 0, "Train/test leakage detected!"
    assert len(val_set & test_set) == 0, "Val/test leakage detected!"

    # Source-level leakage check
    train_sources = set(train_ids)
    val_sources = set(val_ids)
    test_sources = set(test_ids)
    assert len(train_sources & val_sources) == 0, "Source-level train/val leakage!"
    assert len(train_sources & test_sources) == 0, "Source-level train/test leakage!"
    assert len(val_sources & test_sources) == 0, "Source-level val/test leakage!"

    return {
        "train": sorted(train_files),
        "val": sorted(val_files),
        "test": sorted(test_files),
        "metadata": {
            "seed": seed,
            "total_files": len(files),
            "total_sources": n,
            "train_sources": len(train_ids),
            "val_sources": len(val_ids),
            "test_sources": len(test_ids),
            "train_files": len(train_files),
            "val_files": len(val_files),
            "test_files": len(test_files),
            "leakage_check": "PASSED",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Leakage-Free Split Generation")
    parser.add_argument("--image_dir", required=True, help="Directory of images")
    parser.add_argument("--output", default="splits.json", help="Output JSON")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6

    splits = generate_splits(
        args.image_dir, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )

    with open(args.output, "w") as f:
        json.dump(splits, f, indent=2)

    meta = splits["metadata"]
    print(f"Splits generated: {meta['total_files']} files, {meta['total_sources']} sources")
    print(f"  Train: {meta['train_files']} files ({meta['train_sources']} sources)")
    print(f"  Val:   {meta['val_files']} files ({meta['val_sources']} sources)")
    print(f"  Test:  {meta['test_files']} files ({meta['test_sources']} sources)")
    print(f"  Leakage check: {meta['leakage_check']}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
