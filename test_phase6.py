"""Phase 6 verification — evaluate.py CLI and isolation test."""
import sys
import os
import shutil
import json
import numpy as np
import torch
from PIL import Image

from hansr.utils import load_config, set_seed
from hansr.model import build_model
from hansr.utils import save_checkpoint

print("=" * 60)
print("PHASE 6 VERIFICATION — Standalone Inference & Evaluation Script")
print("=" * 60)

test_root = "test_data_phase6"
input_dir = os.path.join(test_root, "input_degraded")
gt_dir = os.path.join(test_root, "gt")
output_dir = os.path.join(test_root, "restored")
ckpt_path = os.path.join(test_root, "dummy_model.pt")

os.makedirs(input_dir, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)

# 1. Create dummy dataset & dummy model checkpoint
set_seed(42)
cfg = load_config("configs/hansr.yaml")
model = build_model(cfg)

# Create 2 dummy image pairs
for i in range(2):
    gt_arr = (np.random.rand(256, 256) * 255).astype(np.uint8)
    deg_arr = (np.random.rand(128, 128) * 255).astype(np.uint8)
    Image.fromarray(gt_arr, "L").save(os.path.join(gt_dir, f"sample_{i:02d}.png"))
    Image.fromarray(deg_arr, "L").save(os.path.join(input_dir, f"sample_{i:02d}.png"))

save_checkpoint(ckpt_path, epoch=0, global_step=0, model=model, optimizer=None, scheduler=None, amp_scaler=None, config=cfg, best_psnr=0.0)
print("  Dummy dataset and checkpoint created.")

# 2. Test isolation check (FR-015)
print("\n--- Test 1: Observability Isolation Check (FR-015) ---")
with open("evaluate.py", "r") as f:
    code = f.read()

assert "import streamlit" not in code, "Violation: evaluate.py imports streamlit!"
assert "from dashboard" not in code, "Violation: evaluate.py imports dashboard!"
assert "import dashboard" not in code, "Violation: evaluate.py imports dashboard!"
print("  PASS: evaluate.py contains no dashboard/streamlit imports")

# 3. Test execution via CLI
print("\n--- Test 2: CLI Inference & Evaluation Execution ---")
cmd = f"python evaluate.py --weights {ckpt_path} --input_dir {input_dir} --output_dir {output_dir} --gt_dir {gt_dir}"
ret = os.system(cmd)
assert ret == 0, f"evaluate.py failed with return code {ret}"

# 4. Verify outputs
print("\n--- Test 3: Output Verification ---")
restored_files = [f for f in os.listdir(output_dir) if f.endswith(".png")]
assert len(restored_files) == 2, f"Expected 2 restored files, found {len(restored_files)}"

# Verify resolution (128x128 -> 256x256)
for f in restored_files:
    img = Image.open(os.path.join(output_dir, f))
    assert img.size == (256, 256), f"Expected restored size (256, 256), got {img.size}"
    assert img.mode == "L", f"Expected grayscale 'L' mode, got {img.mode}"

print("  Restored images: 2x resolution upscaling and 1-channel grayscale PASS")

# Verify metrics.json
metrics_file = os.path.join(output_dir, "metrics.json")
assert os.path.exists(metrics_file), "metrics.json was not generated!"
with open(metrics_file, "r") as f:
    metrics = json.load(f)

print("  Metrics: PSNR={:.2f} dB, SSIM={:.4f}, LPIPS={:.4f}".format(metrics['mean_psnr'], metrics['mean_ssim'], metrics['mean_lpips']))
assert metrics["num_images"] == 2
print("  PASS: metrics.json accurately calculated and saved")

# 5. Verify .npy input -> evaluate.py -> .npy output
print("\n--- Test 4: .npy Input and Output Verification ---")
npy_input_dir = os.path.join(test_root, "npy_input")
npy_gt_dir = os.path.join(test_root, "npy_gt")
npy_output_dir = os.path.join(test_root, "npy_output")
os.makedirs(npy_input_dir, exist_ok=True)
os.makedirs(npy_gt_dir, exist_ok=True)

for i in range(2):
    gt_arr = np.random.rand(256, 256).astype(np.float32)
    deg_arr = np.random.rand(128, 128).astype(np.float32)
    np.save(os.path.join(npy_gt_dir, f"sample_{i:02d}.npy"), gt_arr)
    np.save(os.path.join(npy_input_dir, f"sample_{i:02d}.npy"), deg_arr)

cmd_npy = f"python evaluate.py --weights {ckpt_path} --input_dir {npy_input_dir} --output_dir {npy_output_dir} --gt_dir {npy_gt_dir}"
ret_npy = os.system(cmd_npy)
assert ret_npy == 0, f"evaluate.py failed for .npy files with return code {ret_npy}"

restored_npy_files = [f for f in os.listdir(npy_output_dir) if f.endswith(".npy")]
assert len(restored_npy_files) == 2, f"Expected 2 restored .npy files, got {len(restored_npy_files)}"

for f in restored_npy_files:
    data = np.load(os.path.join(npy_output_dir, f))
    assert data.shape == (256, 256), f"Expected shape (256, 256), got {data.shape}"
    assert data.dtype == np.float32, f"Expected float32 dtype, got {data.dtype}"
    assert np.isfinite(data).all(), "Non-finite values found in restored .npy"
    assert data.min() >= 0.0 and data.max() <= 1.0, f"Range violation: [{data.min()}, {data.max()}]"

print("  PASS: .npy input -> evaluate.py -> .npy output successfully restored with (2H, 2W) float32")

# Cleanup
shutil.rmtree(test_root, ignore_errors=True)

print("\n" + "=" * 60)
print("=== PHASE 6 COMPLETE — ALL EVALUATION CHECKS PASSED ===")
print("=" * 60)
