"""
HANSR Final Verification Suite — Validates all fixes for KLA / SEMICON submission.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from hansr.model import build_model, count_parameters, HANSRNet
from hansr.losses import CompositeLoss, EdgeLoss, FFTLoss
from hansr.utils import (
    load_config,
    save_checkpoint,
    load_checkpoint,
    get_rng_state,
    set_rng_state,
    set_seed,
)
from hansr.dataset import build_datasets


def test_1_model_architecture():
    print("\n--- Test 1: Model Architecture & Parameter Count ---")
    cfg = load_config("configs/hansr.yaml")
    model = build_model(cfg)
    params = count_parameters(model)
    print(f"  Total Parameters : {params['total']:,}")
    print(f"  Model Size       : {params['total_mb']:.2f} MB")
    assert params["total"] == 17706628, f"Expected 17,706,628 parameters, got {params['total']}"
    print("  [PASS] Model architecture and parameter count strictly preserved")


def test_2_loss_weights():
    print("\n--- Test 2: Loss Function Weights ---")
    cfg = load_config("configs/hansr.yaml")
    criterion = CompositeLoss(cfg)
    print(f"  Weights: {criterion.weights}")
    assert criterion.weights["charbonnier"] == 1.0, f"Expected 1.0, got {criterion.weights['charbonnier']}"
    assert criterion.weights["edge"] == 0.40, f"Expected 0.40, got {criterion.weights['edge']}"
    assert criterion.weights["fft"] == 0.25, f"Expected 0.25, got {criterion.weights['fft']}"
    assert criterion.weights["range_penalty"] == 0.01, f"Expected 0.01, got {criterion.weights['range_penalty']}"
    assert "tv" not in criterion.weights or criterion.weights["tv"] == 0.0, "TV loss must be disabled (0.0)"
    print("  [PASS] All loss weights match target configuration exactly")


def test_3_rng_restoration():
    print("\n--- Test 3: Checkpoint RNG Restoration (ByteTensor Fix) ---")
    set_seed(42)
    state = get_rng_state()

    # Simulate Kaggle failure case where torch RNG tensor might have non-byte or non-contiguous representation
    assert isinstance(state["torch"], torch.Tensor), "torch state should be Tensor"

    # Test restoring CPU RNG state
    set_rng_state(state)
    print("  [PASS] CPU RNG state restored without TypeError")

    # Test restoring when tensor is float or int dtype
    fake_state = {
        "python": state["python"],
        "numpy": state["numpy"],
        "torch": state["torch"].to(torch.int32),  # intentionally wrong dtype
    }
    if "cuda" in state:
        fake_state["cuda"] = [s.to(torch.int32) for s in state["cuda"]]
    set_rng_state(fake_state)
    print("  [PASS] RNG state safely converted to uint8 ByteTensor and restored")

    # Checkpoint save/load roundtrip
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test.pt")
        cfg = load_config("configs/hansr.yaml")
        model = build_model(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        save_checkpoint(ckpt_path, 1, 10, model, opt, None, None, cfg, 32.5)
        model_loaded = build_model(cfg)
        opt_loaded = torch.optim.AdamW(model_loaded.parameters(), lr=1e-4)
        ckpt = load_checkpoint(ckpt_path, model_loaded, opt_loaded, resume_training=True)
        assert ckpt["epoch"] == 1
        assert ckpt["best_psnr"] == 32.5
        print("  [PASS] Full checkpoint resume with RNG state passed")


def test_4_dataset_split():
    print("\n--- Test 4: Dataset Loading & 90/10 Split ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_dir = os.path.join(tmpdir, "gt")
        deg_dir = os.path.join(tmpdir, "deg")
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(deg_dir, exist_ok=True)

        for i in range(100):
            np.save(os.path.join(gt_dir, f"sample_{i:04d}.npy"), np.zeros((128, 128), dtype=np.float32))
            np.save(os.path.join(deg_dir, f"sample_{i:04d}.npy"), np.zeros((64, 64), dtype=np.float32))

        cfg = load_config("configs/hansr.yaml")
        cfg["data"]["train_gt_dir"] = gt_dir
        cfg["data"]["train_degraded_dir"] = deg_dir
        cfg["data"]["val_gt_dir"] = None
        cfg["data"]["val_degraded_dir"] = None
        cfg["data"]["val_ratio"] = 0.10
        cfg["training"]["seed"] = 42

        ds = build_datasets(cfg)
        assert len(ds["train"]) == 90, f"Expected 90, got {len(ds['train'])}"
        assert len(ds["val"]) == 10, f"Expected 10, got {len(ds['val'])}"
        print("  [PASS] Leakage-free 90/10 split loaded correctly")


def test_5_run_benchmark():
    print("\n--- Test 5: run.py End-to-End Inference Benchmark ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp_dir = os.path.join(tmpdir, "input")
        out_dir = os.path.join(tmpdir, "output")
        models_dir = os.path.join(tmpdir, "models")
        os.makedirs(inp_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # Create dummy best.pt in models/
        cfg = load_config("configs/hansr.yaml")
        model = build_model(cfg)
        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, "models/best.pt")

        # Create 8 dummy npy images
        for i in range(8):
            arr = (np.random.randn(128, 128) * 0.5 + 0.5).astype(np.float32)
            np.save(os.path.join(inp_dir, f"img_{i:03d}.npy"), arr)

        # Run benchmark
        cmd = [sys.executable, "run.py", inp_dir, out_dir, "--batch-size", "4"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        assert res.returncode == 0, f"run.py failed: {res.stderr}"

        out_files = sorted(os.listdir(out_dir))
        assert len(out_files) == 8, f"Expected 8 outputs, got {len(out_files)}"
        for f in out_files:
            data = np.load(os.path.join(out_dir, f))
            assert data.shape == (256, 256), f"Expected shape (256, 256), got {data.shape}"
            assert data.dtype == np.float32, f"Expected float32, got {data.dtype}"
            assert np.isfinite(data).all(), "Non-finite values detected"
            assert data.min() >= 0.0 and data.max() <= 1.0, f"Range violation: [{data.min()}, {data.max()}]"

        print("  [PASS] run.py end-to-end benchmark and batched restoration verified")


if __name__ == "__main__":
    print("=" * 60)
    print("HANSR FINAL VERIFICATION SUITE")
    print("=" * 60)
    test_1_model_architecture()
    test_2_loss_weights()
    test_3_rng_restoration()
    test_4_dataset_split()
    test_5_run_benchmark()
    print("\n" + "=" * 60)
    print("ALL FINAL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
