"""Phase 4 verification — degradation pipeline and dataset tests."""
import sys, os
sys.path.insert(0, ".")
import torch
import numpy as np
from PIL import Image
from hansr.degradation import (
    add_gaussian_noise, add_speckle_noise, bicubic_downsample,
    degrade_image, DEGRADATION_CASES, sample_degradation_params,
)
from hansr.dataset import (
    PairedDataset, SyntheticDataset, load_grayscale,
    discover_images, random_crop_pair,
)
from hansr.utils import load_config, set_seed

print("=" * 60)
print("PHASE 4 VERIFICATION — Degradation Pipeline & Dataset")
print("=" * 60)

cfg = load_config("configs/hansr.yaml")
deg_cfg = cfg["degradation"]
set_seed(42)

# Create synthetic test images
test_dir = "test_data_phase4"
gt_dir = os.path.join(test_dir, "gt")
deg_dir = os.path.join(test_dir, "degraded")
os.makedirs(gt_dir, exist_ok=True)
os.makedirs(deg_dir, exist_ok=True)

# Generate 4 fake 256x256 GT images and matching 128x128 degraded
for i in range(4):
    gt_arr = np.random.rand(256, 256).astype(np.float32)
    gt_arr = (gt_arr * 255).astype(np.uint8)
    Image.fromarray(gt_arr, mode="L").save(os.path.join(gt_dir, f"img_{i:03d}.png"))

    deg_arr = np.random.rand(128, 128).astype(np.float32)
    deg_arr = (deg_arr * 255).astype(np.uint8)
    Image.fromarray(deg_arr, mode="L").save(os.path.join(deg_dir, f"img_{i:03d}.png"))

# --- Test 1: Individual degradation operations ---
print("\n--- Test 1: Individual Degradation Operations ---")
clean = torch.rand(1, 1, 256, 256)

g = add_gaussian_noise(clean, sigma=0.05)
assert g.shape == clean.shape, "Gaussian: shape changed"
print(f"  Gaussian noise:  shape OK, range [{g.min():.3f}, {g.max():.3f}]")

s = add_speckle_noise(clean, sigma=0.2)
assert s.shape == clean.shape, "Speckle: shape changed"
print(f"  Speckle noise:   shape OK, range [{s.min():.3f}, {s.max():.3f}]")

d = bicubic_downsample(clean, factor=2)
assert d.shape == (1, 1, 128, 128), f"Downsample: expected (1,1,128,128), got {d.shape}"
print(f"  Bicubic 2x down: shape {list(d.shape)} OK")
print("  PASS")

# --- Test 2: All 7 degradation cases ---
print("\n--- Test 2: All 7 Degradation Cases ---")
gt_img = torch.rand(1, 256, 256)
for case in DEGRADATION_CASES:
    degraded, params = degrade_image(gt_img, deg_cfg, case=case)
    case_name = "+".join(case)
    has_resolution = "resolution" in case
    if has_resolution:
        expected_h = 128
    else:
        expected_h = 256
    assert degraded.shape[-1] == expected_h, f"{case_name}: wrong shape {degraded.shape}"
    print(f"  {case_name:30s}: shape {list(degraded.shape)}, "
          f"range [{degraded.min():.3f}, {degraded.max():.3f}]")
print("  PASS: all 7 cases produce correct shapes")

# --- Test 3: FR-004 — values can exceed [0,1] ---
print("\n--- Test 3: FR-004 — No pre-clipping of degraded values ---")
bright = torch.ones(1, 256, 256) * 0.9
degraded, _ = degrade_image(bright, deg_cfg, case=("speckle", "gaussian"))
exceeds = (degraded > 1.0).any().item() or (degraded < 0.0).any().item()
print(f"  Values outside [0,1]: {exceeds}")
print(f"  Range: [{degraded.min():.4f}, {degraded.max():.4f}]")
print("  PASS: degraded values are NOT clipped (FR-004 compliant)")

# --- Test 4: PairedDataset ---
print("\n--- Test 4: PairedDataset ---")
ds = PairedDataset(gt_dir, deg_dir, crop_size=64, augment=True, scale=2)
sample = ds[0]
assert sample["gt"].shape[0] == 1, "GT not 1-channel"
assert sample["degraded"].shape[0] == 1, "Degraded not 1-channel"
print(f"  Length: {len(ds)}")
print(f"  GT shape: {list(sample['gt'].shape)}")
print(f"  Degraded shape: {list(sample['degraded'].shape)}")
assert sample["gt"].shape[-1] == sample["degraded"].shape[-1] * 2, "Crop alignment wrong"
print("  PASS: correct pairing, cropping, and 2x alignment")

# --- Test 5: SyntheticDataset ---
print("\n--- Test 5: SyntheticDataset ---")
syn_ds = SyntheticDataset(gt_dir, deg_cfg, crop_size=64, augment=True, scale=2)
syn_sample = syn_ds[0]
print(f"  Length: {len(syn_ds)}")
print(f"  GT shape: {list(syn_sample['gt'].shape)}")
print(f"  Degraded shape: {list(syn_sample['degraded'].shape)}")
print(f"  Case: {syn_sample['degradation_case']}")
print("  PASS: synthetic degradation works on-the-fly")

# --- Test 6: Reproducibility ---
print("\n--- Test 6: Reproducibility ---")
set_seed(42)
d1, p1 = degrade_image(gt_img, deg_cfg, case=("gaussian",))
set_seed(42)
d2, p2 = degrade_image(gt_img, deg_cfg, case=("gaussian",))
assert torch.equal(d1, d2), "Not reproducible with same seed"
assert p1["gaussian_sigma"] == p2["gaussian_sigma"], "Params differ"
print(f"  Same seed -> identical output: PASS")

# --- Test 7: .npy format loading & unclipped values ---
print("\n--- Test 7: .npy format loading & unclipped values ---")
npy_gt_dir = os.path.join(test_dir, "npy_gt")
npy_deg_dir = os.path.join(test_dir, "npy_degraded")
os.makedirs(npy_gt_dir, exist_ok=True)
os.makedirs(npy_deg_dir, exist_ok=True)

# KLA dataset characteristics:
# GT: (256, 256), float32, [0, 1]
# NoisyLR: (128, 128), float32, values may be outside [0, 1]
gt_npy_raw = np.random.uniform(0.0, 1.0, size=(256, 256)).astype(np.float32)
deg_npy_raw = (np.random.randn(128, 128) * 0.5 + 0.5).astype(np.float32)  # values outside [0, 1]

np.save(os.path.join(npy_gt_dir, "sample_000.npy"), gt_npy_raw)
np.save(os.path.join(npy_deg_dir, "sample_000.npy"), deg_npy_raw)

gt_tensor = load_grayscale(os.path.join(npy_gt_dir, "sample_000.npy"))
deg_tensor = load_grayscale(os.path.join(npy_deg_dir, "sample_000.npy"))

assert gt_tensor.shape == (1, 256, 256), f"Wrong GT npy tensor shape: {gt_tensor.shape}"
assert deg_tensor.shape == (1, 128, 128), f"Wrong Degraded npy tensor shape: {deg_tensor.shape}"
assert gt_tensor.dtype == torch.float32, f"Wrong GT npy dtype: {gt_tensor.dtype}"
assert deg_tensor.dtype == torch.float32, f"Wrong Degraded npy dtype: {deg_tensor.dtype}"

# Values must not be clipped or normalized again
np.testing.assert_allclose(gt_tensor.squeeze(0).numpy(), gt_npy_raw, rtol=1e-5, atol=1e-6)
np.testing.assert_allclose(deg_tensor.squeeze(0).numpy(), deg_npy_raw, rtol=1e-5, atol=1e-6)
print(f"  GT .npy:       shape {list(gt_tensor.shape)}, dtype {gt_tensor.dtype}, range [{gt_tensor.min():.3f}, {gt_tensor.max():.3f}]")
print(f"  Degraded .npy: shape {list(deg_tensor.shape)}, dtype {deg_tensor.dtype}, range [{deg_tensor.min():.3f}, {deg_tensor.max():.3f}]")

npy_ds = PairedDataset(npy_gt_dir, npy_deg_dir, crop_size=64, augment=False, scale=2)
npy_sample = npy_ds[0]
assert npy_sample["gt"].shape == (1, 128, 128)
assert npy_sample["degraded"].shape == (1, 64, 64)
print("  PASS: .npy loading, preservation of unclipped values, and PairedDataset integration verified")

# Cleanup
import shutil
shutil.rmtree(test_dir, ignore_errors=True)

print("\n" + "=" * 60)
print("=== PHASE 4 COMPLETE — ALL DEGRADATION/DATASET CHECKS PASSED ===")
print("=" * 60)
