"""
Dataset Verification Gate (FR-009)

Runs before training to validate:
  1. Pairing completeness (every GT has a degraded match)
  2. Decodability (can Pillow load each file?)
  3. Channel count (single-channel grayscale, FR-003)
  4. Dimension relationships (degraded H*2 == GT H, FR-002)
  5. GT range compliance ([0, 1])
  6. Duplicate/ambiguous IDs

Outputs: results/dataset_verification.json
Training is blocked if the gate fails.

Usage:
    python scripts/verify_dataset.py --gt_dir data/train/gt --degraded_dir data/train/degraded
    python scripts/verify_dataset.py --config configs/hansr.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image
from hansr.utils import load_config, ensure_dir


def verify_dataset(gt_dir: str, degraded_dir: str, scale: int = 2) -> dict:
    """
    Run all verification checks on a paired dataset.

    Returns:
        Report dict with 'passed', 'errors', 'warnings', and 'stats'.
    """
    report = {
        "passed": True,
        "gt_dir": gt_dir,
        "degraded_dir": degraded_dir,
        "scale": scale,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    # --- Check 1: Directories exist ---
    kaggle_gt = "/kaggle/input/datasets/jhansiranimajhi/kla-dataset/train/GT"
    kaggle_deg = "/kaggle/input/datasets/jhansiranimajhi/kla-dataset/train/NoisyLR"
    if not os.path.isdir(gt_dir) and os.path.isdir(kaggle_gt):
        gt_dir = kaggle_gt
        degraded_dir = kaggle_deg
        report["gt_dir"] = gt_dir
        report["degraded_dir"] = degraded_dir

    if not os.path.isdir(gt_dir):
        report["errors"].append(f"GT directory not found: {gt_dir}")
        report["passed"] = False
        return report
    if not os.path.isdir(degraded_dir):
        report["errors"].append(f"Degraded directory not found: {degraded_dir}")
        report["passed"] = False
        return report

    # --- Check 2: Discover files ---
    exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".npy"}
    gt_files = sorted([f for f in os.listdir(gt_dir)
                       if Path(f).suffix.lower() in exts])
    deg_files = sorted([f for f in os.listdir(degraded_dir)
                        if Path(f).suffix.lower() in exts])

    report["stats"]["gt_count"] = len(gt_files)
    report["stats"]["degraded_count"] = len(deg_files)

    if len(gt_files) == 0:
        report["errors"].append("No GT images found")
        report["passed"] = False
        return report

    # --- Check 3: Pairing completeness ---
    gt_stems = {Path(f).stem for f in gt_files}
    deg_stems = {Path(f).stem for f in deg_files}
    unpaired_gt = gt_stems - deg_stems
    unpaired_deg = deg_stems - gt_stems

    if unpaired_gt:
        report["errors"].append(f"GT images without degraded pair: {sorted(unpaired_gt)[:10]}")
        report["passed"] = False
    if unpaired_deg:
        report["warnings"].append(f"Degraded images without GT pair: {sorted(unpaired_deg)[:10]}")

    # --- Check 4: Duplicate stems ---
    gt_stem_list = [Path(f).stem for f in gt_files]
    dupes = [s for s in set(gt_stem_list) if gt_stem_list.count(s) > 1]
    if dupes:
        report["errors"].append(f"Duplicate GT filenames (ambiguous): {dupes[:10]}")
        report["passed"] = False

    report["stats"]["paired_count"] = len(gt_stems & deg_stems)

    # --- Check 5: Per-file validation ---
    decode_errors = []
    channel_errors = []
    dim_errors = []
    range_warnings = []

    paired_stems = sorted(gt_stems & deg_stems)
    gt_lookup = {Path(f).stem: f for f in gt_files}
    deg_lookup = {Path(f).stem: f for f in deg_files}

    for stem in paired_stems:
        gt_path = os.path.join(gt_dir, gt_lookup[stem])
        deg_path = os.path.join(degraded_dir, deg_lookup[stem])

        # Decodability
        try:
            if gt_path.lower().endswith(".npy"):
                gt_arr = np.load(gt_path)
            else:
                gt_img = Image.open(gt_path)
                gt_arr = np.array(gt_img)
        except Exception as e:
            decode_errors.append(f"GT {stem}: {e}")
            continue

        try:
            if deg_path.lower().endswith(".npy"):
                deg_arr = np.load(deg_path)
            else:
                deg_img = Image.open(deg_path)
                deg_arr = np.array(deg_img)
        except Exception as e:
            decode_errors.append(f"Degraded {stem}: {e}")
            continue

        # Squeeze leading channel dim if (1, H, W)
        if gt_arr.ndim == 3 and gt_arr.shape[0] == 1:
            gt_arr = gt_arr.squeeze(0)
        if deg_arr.ndim == 3 and deg_arr.shape[0] == 1:
            deg_arr = deg_arr.squeeze(0)

        # Grayscale check (FR-003)
        if gt_arr.ndim != 2:
            channel_errors.append(f"GT {stem}: ndim={gt_arr.ndim}, expected 2 (grayscale)")
        if deg_arr.ndim != 2:
            channel_errors.append(f"Degraded {stem}: ndim={deg_arr.ndim}, expected 2")

        # Dimension relationship (FR-002)
        if gt_arr.ndim == 2 and deg_arr.ndim == 2:
            expected_gt_h = deg_arr.shape[0] * scale
            expected_gt_w = deg_arr.shape[1] * scale
            if gt_arr.shape[0] != expected_gt_h or gt_arr.shape[1] != expected_gt_w:
                dim_errors.append(
                    f"{stem}: GT {gt_arr.shape} vs Degraded {deg_arr.shape} "
                    f"(expected {scale}x relationship)"
                )

        # GT range compliance
        if gt_arr.dtype == np.uint8:
            pass  # [0, 255] is fine, normalized at load time
        else:
            gt_min, gt_max = gt_arr.min(), gt_arr.max()
            if gt_max > 1.0 or gt_min < 0.0:
                range_warnings.append(f"GT {stem}: range [{gt_min:.3f}, {gt_max:.3f}]")

    if decode_errors:
        report["errors"].extend(decode_errors[:10])
        report["passed"] = False
    if channel_errors:
        report["errors"].extend(channel_errors[:10])
        report["passed"] = False
    if dim_errors:
        report["errors"].extend(dim_errors[:10])
        report["passed"] = False
    if range_warnings:
        report["warnings"].extend(range_warnings[:10])

    report["stats"]["decode_errors"] = len(decode_errors)
    report["stats"]["channel_errors"] = len(channel_errors)
    report["stats"]["dimension_errors"] = len(dim_errors)
    report["stats"]["range_warnings"] = len(range_warnings)

    return report


def main():
    parser = argparse.ArgumentParser(description="Dataset Verification Gate (FR-009)")
    parser.add_argument("--gt_dir", default=None, help="GT image directory")
    parser.add_argument("--degraded_dir", default=None, help="Degraded image directory")
    parser.add_argument("--config", default=None, help="YAML config (uses data paths)")
    parser.add_argument("--output", default="results/dataset_verification.json")
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        gt_dir = cfg["data"]["train_gt_dir"]
        deg_dir = cfg["data"]["train_degraded_dir"]
        scale = cfg["model"].get("upscale_factor", 2)
    elif args.gt_dir and args.degraded_dir:
        gt_dir = args.gt_dir
        deg_dir = args.degraded_dir
        scale = args.scale
    else:
        print("ERROR: Provide either --config or both --gt_dir and --degraded_dir")
        sys.exit(1)

    print(f"Verifying dataset: {gt_dir} / {deg_dir} (scale={scale})")
    report = verify_dataset(gt_dir, deg_dir, scale)

    # Save report
    ensure_dir(os.path.dirname(args.output))
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\nVerification {'PASSED' if report['passed'] else 'FAILED'}")
    print(f"  Stats: {json.dumps(report['stats'], indent=4)}")
    if report["errors"]:
        print(f"  Errors ({len(report['errors'])}):")
        for e in report["errors"]:
            print(f"    - {e}")
    if report["warnings"]:
        print(f"  Warnings ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"    - {w}")
    print(f"\nReport saved to: {args.output}")

    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
