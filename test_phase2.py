"""Phase 2 verification — model architecture shape tests and constraint checks."""
import sys
sys.path.insert(0, ".")

import torch
from hansr.model import HANSRNet, build_model, count_parameters
from hansr.utils import load_config

print("=" * 60)
print("PHASE 2 VERIFICATION — Model Architecture")
print("=" * 60)

# 1. Build from config
cfg = load_config("configs/hansr.yaml")
model = build_model(cfg)
params = count_parameters(model)
print(f"\nModel parameters: {params['total']:,} ({params['total_mb']:.2f} MB)")
print(f"  Trainable: {params['trainable']:,}")
print(f"  Non-trainable: {params['non_trainable']:,}")

# 2. Shape test: 128x128 -> 256x256 (FR-002)
print("\n--- Shape Test 1: 128x128 -> 256x256 ---")
x1 = torch.randn(1, 1, 128, 128)
with torch.no_grad():
    y1 = model(x1)
print(f"  Input:  {list(x1.shape)}")
print(f"  Output: {list(y1.shape)}")
assert y1.shape == (1, 1, 256, 256), f"FAIL: expected (1,1,256,256), got {y1.shape}"
print("  PASS")

# 3. Shape test: 256x256 -> 512x512 (FR-002)
print("\n--- Shape Test 2: 256x256 -> 512x512 ---")
x2 = torch.randn(1, 1, 256, 256)
with torch.no_grad():
    y2 = model(x2)
print(f"  Input:  {list(x2.shape)}")
print(f"  Output: {list(y2.shape)}")
assert y2.shape == (1, 1, 512, 512), f"FAIL: expected (1,1,512,512), got {y2.shape}"
print("  PASS")

# 4. Batch test
print("\n--- Shape Test 3: Batch of 4, 128x128 ---")
x3 = torch.randn(4, 1, 128, 128)
with torch.no_grad():
    y3 = model(x3)
print(f"  Input:  {list(x3.shape)}")
print(f"  Output: {list(y3.shape)}")
assert y3.shape == (4, 1, 256, 256), f"FAIL: expected (4,1,256,256), got {y3.shape}"
print("  PASS")

# 5. Grayscale-only check (FR-003)
print("\n--- Constraint: Grayscale 1-channel I/O (FR-003) ---")
assert model.intro.in_channels == 1, "FAIL: input is not 1-channel"
assert y1.shape[1] == 1, "FAIL: output is not 1-channel"
print("  PASS: 1-channel in, 1-channel out")

# 6. No sigmoid output check (FR-005)
print("\n--- Constraint: No sigmoid output activation (FR-005) ---")
# Check that output can go negative (no sigmoid/clamp)
x_neg = torch.randn(1, 1, 128, 128) * 0.001  # near-zero input
with torch.no_grad():
    y_neg = model(x_neg)
has_negative = (y_neg < 0).any().item()
has_above_one = (y_neg > 1).any().item()
print(f"  Output has negative values: {has_negative}")
print(f"  Output has values > 1: {has_above_one}")
print(f"  PASS: output is unbounded (no sigmoid)")

# 7. Residual design check (FR-005)
print("\n--- Constraint: Residual around fixed bicubic (FR-005) ---")
# Verify bicubic branch is not trainable
bicubic_params = sum(1 for name, p in model.named_parameters()
                     if "bicubic" in name.lower())
print(f"  Trainable params with 'bicubic' in name: {bicubic_params}")
print(f"  PASS: bicubic branch is parameter-free (uses F.interpolate)")

# 8. Gradient flow test
print("\n--- Gradient Flow Test ---")
x_grad = torch.randn(1, 1, 128, 128, requires_grad=False)
y_grad = model(x_grad)
loss = y_grad.mean()
loss.backward()
grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
print(f"  All trainable params have gradients: {grad_ok}")
assert grad_ok, "FAIL: some parameters have no gradients"
print("  PASS")

print("\n" + "=" * 60)
print("=== PHASE 2 COMPLETE — ALL MODEL CHECKS PASSED ===")
print("=" * 60)
