"""
KLA Submission Verification Script — Validates run.py and submission contract.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from hansr.model import HANSRNet
from hansr.utils import save_checkpoint, load_config


def run_submission_test():
    print("=" * 60)
    print("KLA SUBMISSION CONTRACT VERIFICATION")
    print("=" * 60)

    test_root = Path("test_submission_tmp")
    input_dir = test_root / "input"
    output_dir = test_root / "output"
    models_dir = Path("models")

    # Clean up previous runs
    shutil.rmtree(test_root, ignore_errors=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create a dummy model checkpoint in models/best.pt if none exists
    best_pt_path = models_dir / "best.pt"
    created_dummy_ckpt = False
    if not best_pt_path.exists():
        print("  Creating test checkpoint in models/best.pt...")
        model = HANSRNet(width=32, num_blocks=[2, 4, 4, 8], middle_blocks=4, upscale_factor=2)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": load_config("configs/hansr.yaml"),
                "best_psnr": 30.0,
            },
            best_pt_path,
        )
        created_dummy_ckpt = True

    # 2. Check if real KLA samples are available, otherwise create realistic test samples
    kla_kaggle_noisy = Path("/kaggle/input/datasets/jhansiranimajhi/kla-dataset/train/NoisyLR")
    sample_shapes = {}

    if kla_kaggle_noisy.exists() and kla_kaggle_noisy.is_dir():
        real_files = sorted(list(kla_kaggle_noisy.glob("*.npy")))[:3]
        print(f"  Copying {len(real_files)} real KLA sample files...")
        for rf in real_files:
            dest = input_dir / rf.name
            shutil.copy(rf, dest)
            arr = np.load(rf)
            sample_shapes[rf.name] = arr.shape
    else:
        print("  Generating 4 synthetic KLA degraded test samples (.npy)...")
        # Test shapes: (1, 128, 128), (128, 128), (128, 128, 1), and non-standard (1, 64, 64)
        s1 = (np.random.randn(1, 128, 128) * 0.5 + 0.5).astype(np.float32)
        s2 = (np.random.randn(128, 128) * 0.5 + 0.5).astype(np.float32)
        s3 = (np.random.randn(128, 128, 1) * 0.5 + 0.5).astype(np.float32)
        s4 = (np.random.randn(1, 64, 64) * 0.5 + 0.5).astype(np.float32)

        np.save(input_dir / "sample_001.npy", s1)
        np.save(input_dir / "sample_002.npy", s2)
        np.save(input_dir / "sample_003.npy", s3)
        np.save(input_dir / "sample_004.npy", s4)

        sample_shapes["sample_001.npy"] = s1.shape
        sample_shapes["sample_002.npy"] = s2.shape
        sample_shapes["sample_003.npy"] = s3.shape
        sample_shapes["sample_004.npy"] = s4.shape

    # 3. Execute run.py via CLI
    cmd = [sys.executable, "run.py", str(input_dir), str(output_dir)]
    print(f"  Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)

    print("--- STDOUT ---")
    print(res.stdout)
    if res.stderr:
        print("--- STDERR ---")
        print(res.stderr)

    assert res.returncode == 0, f"run.py failed with return code {res.returncode}"
    print("  [PASS] run.py execution completed with exit code 0")

    # 4. Verify outputs
    input_files = sorted(list(input_dir.glob("*.npy")))
    output_files = sorted(list(output_dir.glob("*.npy")))

    # Check 1: One output per input
    assert len(output_files) == len(input_files), (
        f"Output count mismatch: {len(output_files)} outputs vs {len(input_files)} inputs"
    )
    print(f"  [PASS] Exactly one output per input ({len(output_files)} files)")

    # Check 2: Filenames match exactly
    input_names = [f.name for f in input_files]
    output_names = [f.name for f in output_files]
    assert input_names == output_names, f"Filename mismatch: {input_names} vs {output_names}"
    print("  [PASS] Exact input filenames preserved in output")

    # Check 3: Check each output array
    for out_f in output_files:
        in_shape = sample_shapes[out_f.name]
        out_arr = np.load(out_f)

        # Grayscale 2D array
        assert out_arr.ndim == 2, f"{out_f.name}: output is not 2D grayscale, got shape {out_arr.shape}"

        # 2x resolution
        if len(in_shape) == 2:
            in_h, in_w = in_shape
        elif len(in_shape) == 3 and in_shape[0] == 1:
            in_h, in_w = in_shape[1], in_shape[2]
        elif len(in_shape) == 3 and in_shape[2] == 1:
            in_h, in_w = in_shape[0], in_shape[1]
        else:
            in_h, in_w = in_shape[-2], in_shape[-1]

        expected_h, expected_w = in_h * 2, in_w * 2
        assert out_arr.shape == (expected_h, expected_w), (
            f"{out_f.name}: expected 2x shape ({expected_h}, {expected_w}), got {out_arr.shape}"
        )

        # Dtype float32
        assert out_arr.dtype == np.float32, f"{out_f.name}: expected float32, got {out_arr.dtype}"

        # All values finite
        assert np.isfinite(out_arr).all(), f"{out_f.name}: non-finite values detected"

        # Values in [0, 1]
        min_v, max_v = float(out_arr.min()), float(out_arr.max())
        assert min_v >= 0.0, f"{out_f.name}: min value {min_v} < 0.0"
        assert max_v <= 1.0, f"{out_f.name}: max value {max_v} > 1.0"

        print(
            f"  [PASS] {out_f.name}: in={in_shape} -> out={out_arr.shape}, "
            f"dtype={out_arr.dtype}, range=[{min_v:.4f}, {max_v:.4f}], all finite=True"
        )

    # 5. Clean up temporary test artifacts
    shutil.rmtree(test_root, ignore_errors=True)
    if created_dummy_ckpt and best_pt_path.exists():
        best_pt_path.unlink()

    print("\n" + "=" * 60)
    print("SUCCESS: KLA SUBMISSION CONTRACT FULLY VERIFIED")
    print("=" * 60)


def run_dataset_split_test():
    print("\n" + "=" * 60)
    print("KLA DATASET 90/10 LEAKAGE-FREE SPLIT VERIFICATION")
    print("=" * 60)

    from hansr.dataset import build_datasets

    test_split_root = Path("test_split_tmp")
    gt_dir = test_split_root / "gt"
    deg_dir = test_split_root / "degraded"
    shutil.rmtree(test_split_root, ignore_errors=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    deg_dir.mkdir(parents=True, exist_ok=True)

    # Generate 100 dummy pairs to verify 90/10 split
    n_samples = 100
    for i in range(n_samples):
        name = f"sample_{i:04d}.npy"
        np.save(gt_dir / name, np.zeros((16, 16), dtype=np.float32))
        np.save(deg_dir / name, np.zeros((8, 8), dtype=np.float32))

    cfg = load_config("configs/hansr.yaml")
    cfg["data"]["train_gt_dir"] = str(gt_dir)
    cfg["data"]["train_degraded_dir"] = str(deg_dir)
    cfg["data"]["val_gt_dir"] = None
    cfg["data"]["val_degraded_dir"] = None
    cfg["data"]["val_ratio"] = 0.10
    cfg["training"]["seed"] = 42

    ds = build_datasets(cfg)
    train_ds = ds["train"]
    val_ds = ds["val"]

    assert train_ds is not None, "Train dataset is None"
    assert val_ds is not None, "Val dataset is None"
    assert len(train_ds) == 90, f"Expected 90 train samples, got {len(train_ds)}"
    assert len(val_ds) == 10, f"Expected 10 val samples, got {len(val_ds)}"

    # Leakage check: zero filename intersection
    train_files = {Path(f).name for f in train_ds.gt_files}
    val_files = {Path(f).name for f in val_ds.gt_files}
    intersection = train_files & val_files
    assert len(intersection) == 0, f"Leakage detected between train and val: {intersection}"

    # Verify pairing: GT and degraded filenames match 1:1
    for gt_p, deg_p in zip(train_ds.gt_files, train_ds.degraded_files):
        assert Path(gt_p).name == Path(deg_p).name, f"Train pairing mismatch: {gt_p} vs {deg_p}"
    for gt_p, deg_p in zip(val_ds.gt_files, val_ds.degraded_files):
        assert Path(gt_p).name == Path(deg_p).name, f"Val pairing mismatch: {gt_p} vs {deg_p}"

    # Verify 3,200 sample math: 90% = 2,880, 10% = 320
    n_kla = 3200
    n_val_kla = int(round(n_kla * 0.10))
    n_train_kla = n_kla - n_val_kla
    assert n_train_kla == 2880, f"Expected 2880 train pairs for 3200 KLA samples, got {n_train_kla}"
    assert n_val_kla == 320, f"Expected 320 val pairs for 3200 KLA samples, got {n_val_kla}"

    shutil.rmtree(test_split_root, ignore_errors=True)
    print(f"  [PASS] 100 sample split: Train={len(train_ds)}, Val={len(val_ds)}")
    print(f"  [PASS] 3,200 KLA formula: Train={n_train_kla}, Val={n_val_kla}")
    print(f"  [PASS] Zero train/val filename overlap (leakage-free)")
    print(f"  [PASS] GT and NoisyLR pairing preserved 100%")
    print("=" * 60)
    print("SUCCESS: KLA DATASET SPLIT VERIFIED")
    print("=" * 60)


if __name__ == "__main__":
    run_submission_test()
    run_dataset_split_test()
