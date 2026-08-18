"""Phase 5 verification — training pipeline, dataset verification, splits, metrics."""
import sys, os, shutil, json
sys.path.insert(0, ".")
import numpy as np
import torch
from PIL import Image
from hansr.utils import load_config, set_seed, setup_logging
from hansr.model import build_model
from hansr.losses import CompositeLoss
from hansr.metrics import compute_psnr, compute_ssim

setup_logging()

print("=" * 60)
print("PHASE 5 VERIFICATION — Training Pipeline")
print("=" * 60)

# --- Setup: create synthetic paired dataset ---
cfg = load_config("configs/hansr.yaml")
test_root = "test_data_phase5"
for split in ["train", "val"]:
    gt_d = os.path.join(test_root, split, "gt")
    deg_d = os.path.join(test_root, split, "degraded")
    os.makedirs(gt_d, exist_ok=True)
    os.makedirs(deg_d, exist_ok=True)
    n = 8 if split == "train" else 4
    for i in range(n):
        gt = (np.random.rand(256, 256) * 255).astype(np.uint8)
        deg = (np.random.rand(128, 128) * 255).astype(np.uint8)
        Image.fromarray(gt, "L").save(os.path.join(gt_d, f"img_{i:03d}.png"))
        Image.fromarray(deg, "L").save(os.path.join(deg_d, f"img_{i:03d}.png"))

# Patch config for test data
cfg["data"]["train_gt_dir"] = f"{test_root}/train/gt"
cfg["data"]["train_degraded_dir"] = f"{test_root}/train/degraded"
cfg["data"]["val_gt_dir"] = f"{test_root}/val/gt"
cfg["data"]["val_degraded_dir"] = f"{test_root}/val/degraded"
cfg["data"]["num_workers"] = 0
cfg["training"]["batch_size"] = 2
cfg["training"]["patch_size"] = 64

# --- Test 1: Metrics ---
print("\n--- Test 1: Metrics (PSNR / SSIM) ---")
set_seed(42)
a = torch.rand(1, 1, 128, 128)
b = a.clone()
psnr_same = compute_psnr(a, b)
ssim_same = compute_ssim(a, b)
print(f"  PSNR (identical): {psnr_same:.1f} dB")
print(f"  SSIM (identical): {ssim_same:.4f}")
assert psnr_same >= 90, "PSNR should be very high for identical images"
assert ssim_same > 0.99, "SSIM should be ~1.0 for identical images"

c = a + torch.randn_like(a) * 0.1
psnr_diff = compute_psnr(a, c)
ssim_diff = compute_ssim(a, c)
print(f"  PSNR (noisy):     {psnr_diff:.1f} dB")
print(f"  SSIM (noisy):     {ssim_diff:.4f}")
assert psnr_diff < psnr_same, "Noisy should have lower PSNR"
print("  PASS")

# --- Test 2: Dataset Verification Gate ---
print("\n--- Test 2: Dataset Verification Gate (FR-009) ---")
from scripts.verify_dataset import verify_dataset
report = verify_dataset(
    f"{test_root}/train/gt", f"{test_root}/train/degraded", scale=2
)
print(f"  Verification: {'PASSED' if report['passed'] else 'FAILED'}")
print(f"  Stats: {report['stats']}")
assert report["passed"], f"Verification should pass: {report['errors']}"
print("  PASS")

# --- Test 3: Split Generation ---
print("\n--- Test 3: Leakage-Free Splits (FR-010) ---")
from scripts.generate_splits import generate_splits, extract_source_id
splits = generate_splits(f"{test_root}/train/gt", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
meta = splits["metadata"]
print(f"  Train: {meta['train_files']}, Val: {meta['val_files']}, Test: {meta['test_files']}")
print(f"  Leakage check: {meta['leakage_check']}")
assert meta["leakage_check"] == "PASSED"
train_set = set(splits["train"])
val_set = set(splits["val"])
test_set = set(splits["test"])
assert len(train_set & val_set) == 0, "Leakage!"
assert len(train_set & test_set) == 0, "Leakage!"
assert len(val_set & test_set) == 0, "Leakage!"
print("  PASS: zero overlap between splits (standard images)")

# --- Test 3b: Leakage-Free Splits with .npy files ---
print("\n--- Test 3b: Leakage-Free Splits with .npy files ---")
npy_split_dir = os.path.join(test_root, "npy_split_test")
os.makedirs(npy_split_dir, exist_ok=True)
npy_specimens = ["wafer_A", "wafer_B", "wafer_C", "wafer_D", "wafer_E", "wafer_F"]
total_npy_files = []
for spec in npy_specimens:
    for patch_idx in range(3):
        fname = f"{spec}_patch_{patch_idx:03d}.npy"
        np.save(os.path.join(npy_split_dir, fname), np.zeros((16, 16), dtype=np.float32))
        total_npy_files.append(fname)

npy_splits = generate_splits(npy_split_dir, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42)
npy_meta = npy_splits["metadata"]
print(f"  .npy splits: Total {npy_meta['total_files']} files ({npy_meta['total_sources']} sources)")
print(f"    Train: {npy_meta['train_files']} files ({npy_meta['train_sources']} sources)")
print(f"    Val:   {npy_meta['val_files']} files ({npy_meta['val_sources']} sources)")
print(f"    Test:  {npy_meta['test_files']} files ({npy_meta['test_sources']} sources)")

# 1. All .npy files are included exactly once
all_split_npy = npy_splits["train"] + npy_splits["val"] + npy_splits["test"]
assert len(all_split_npy) == len(total_npy_files), "Not all .npy files included"
assert set(all_split_npy) == set(total_npy_files), "Mismatch in split .npy filenames"

# 2. Train/val/test have no overlap
assert len(set(npy_splits["train"]) & set(npy_splits["val"])) == 0, ".npy train/val overlap!"
assert len(set(npy_splits["train"]) & set(npy_splits["test"])) == 0, ".npy train/test overlap!"
assert len(set(npy_splits["val"]) & set(npy_splits["test"])) == 0, ".npy val/test overlap!"

# 3. Source-level leakage check passes
train_src = {extract_source_id(f) for f in npy_splits["train"]}
val_src = {extract_source_id(f) for f in npy_splits["val"]}
test_src = {extract_source_id(f) for f in npy_splits["test"]}
assert len(train_src & val_src) == 0, "Source leakage between train and val!"
assert len(train_src & test_src) == 0, "Source leakage between train and test!"
assert len(val_src & test_src) == 0, "Source leakage between val and test!"

# 4. JSON dump / load verification
splits_json_path = os.path.join(test_root, "test_splits.json")
with open(splits_json_path, "w") as f:
    json.dump(npy_splits, f, indent=2)
with open(splits_json_path, "r") as f:
    loaded_splits = json.load(f)
assert loaded_splits["train"] == npy_splits["train"]
assert loaded_splits["val"] == npy_splits["val"]
assert loaded_splits["test"] == npy_splits["test"]
print(f"  Leakage check: {npy_meta['leakage_check']}")
print("  PASS: .npy splits generated correctly with zero source leakage")

# --- Test 4: Training mini-loop (2 epochs) ---
print("\n--- Test 4: Training Mini-Loop (2 epochs) ---")
from hansr.dataset import build_datasets
from torch.utils.data import DataLoader
from train import train_one_epoch, validate, build_optimizer, build_scheduler

device = torch.device("cpu")
model = build_model(cfg).to(device)
criterion = CompositeLoss(cfg).to(device)
optimizer = build_optimizer(model, cfg)
scheduler = build_scheduler(optimizer, cfg)

datasets = build_datasets(cfg)
train_loader = DataLoader(datasets["train"], batch_size=2, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(datasets["val"], batch_size=1, shuffle=False, num_workers=0)

global_step = 0
for epoch in range(2):
    loss, breakdown, global_step = train_one_epoch(
        model, train_loader, criterion, optimizer, None, device, cfg, epoch, None, global_step
    )
    print(f"  Epoch {epoch}: loss={loss:.4f}, step={global_step}")
    scheduler.step()

val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device, cfg)
print(f"  Val: loss={val_loss:.4f}, psnr={val_psnr:.2f}, ssim={val_ssim:.4f}")
assert val_psnr > 0, "PSNR should be positive"
print("  PASS: training loop runs without errors")

# --- Test 5: Checkpoint save/load ---
print("\n--- Test 5: Checkpoint Save/Load (FR-011) ---")
from hansr.utils import save_checkpoint, load_checkpoint
ckpt_path = os.path.join(test_root, "test_ckpt.pt")
save_checkpoint(ckpt_path, 1, global_step, model, optimizer, scheduler, None, cfg, val_psnr)
assert os.path.exists(ckpt_path), "Checkpoint not saved"

model2 = build_model(cfg).to(device)
ckpt = load_checkpoint(ckpt_path, model2, device=device)
assert ckpt["epoch"] == 1
assert ckpt["best_psnr"] == val_psnr

# Verify weights match
for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
    assert torch.equal(p1, p2), f"Weight mismatch: {n1}"
print(f"  Saved & loaded checkpoint, epoch={ckpt['epoch']}, psnr={ckpt['best_psnr']:.2f}")
print("  Weights match: PASS")

# Cleanup
shutil.rmtree(test_root, ignore_errors=True)

print("\n" + "=" * 60)
print("=== PHASE 5 COMPLETE — ALL TRAINING PIPELINE CHECKS PASSED ===")
print("=" * 60)
