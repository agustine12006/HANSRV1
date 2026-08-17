"""Phase 1 verification script — validates config loading, seeding, device detection."""
import sys
sys.path.insert(0, ".")

from hansr.utils import load_config, set_seed, get_device, setup_logging

setup_logging()

# 1. Config loading
cfg = load_config("configs/hansr.yaml")
print(f"Model width: {cfg['model']['width']}")
print(f"Num blocks: {cfg['model']['num_blocks']}")
print(f"Loss terms: {list(cfg['loss'].keys())}")
print(f"Seed: {cfg['training']['seed']}")
print(f"Degradation gaussian range: {cfg['degradation']['gaussian']['sigma_range']}")
print(f"Degradation speckle range: {cfg['degradation']['speckle']['sigma_range']}")

# 2. Seeding
set_seed(cfg["training"]["seed"])
import torch
t1 = torch.randn(3)
set_seed(cfg["training"]["seed"])
t2 = torch.randn(3)
assert torch.equal(t1, t2), "Seeding failed — tensors not equal!"
print(f"Seed reproducibility: PASS (t1={t1.tolist()}, t2={t2.tolist()})")

# 3. Device detection
dev = get_device()
print(f"Device: {dev}")

# 4. Verify project structure
import os
expected_files = [
    "configs/hansr.yaml",
    "hansr/__init__.py",
    "hansr/utils.py",
    "dashboard/__init__.py",
    "scripts/verify_dataset.py",
    "scripts/generate_splits.py",
    "scripts/generate_degraded.py",
    "docs/model_card.md",
    "requirements.txt",
    "README.md",
    ".gitignore",
]
all_ok = True
for f in expected_files:
    exists = os.path.exists(f)
    status = "OK" if exists else "MISSING"
    if not exists:
        all_ok = False
    print(f"  [{status}] {f}")

if all_ok:
    print("\n=== PHASE 1 COMPLETE — ALL CHECKS PASSED ===")
else:
    print("\n=== PHASE 1 INCOMPLETE — SOME FILES MISSING ===")
    sys.exit(1)
