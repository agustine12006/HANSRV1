"""Phase 3 verification — loss function tests (FR-007, FR-006)."""
import sys
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
from hansr.losses import (
    CharbonnierLoss, EdgeLoss, FFTLoss,
    RangePenaltyLoss, TVLoss, CompositeLoss,
)
from hansr.utils import load_config

print("=" * 60)
print("PHASE 3 VERIFICATION — Five-Term Composite Loss")
print("=" * 60)

B, C, H, W = 2, 1, 256, 256
pred = torch.randn(B, C, H, W) * 0.5 + 0.5
gt = torch.randn(B, C, H, W) * 0.5 + 0.5

# --- Individual terms ---
print("\n--- Individual Loss Terms ---")
tests = [
    ("Charbonnier", CharbonnierLoss()),
    ("Edge (Sobel)", EdgeLoss()),
    ("FFT Magnitude", FFTLoss()),
    ("Range Penalty", RangePenaltyLoss()),
    ("Total Variation", TVLoss()),
]
for name, loss_fn in tests:
    val = loss_fn(pred, gt)
    assert val.ndim == 0, f"{name}: expected scalar, got shape {val.shape}"
    assert torch.isfinite(val), f"{name}: non-finite value {val.item()}"
    assert val.requires_grad or val.item() >= 0, f"{name}: negative loss"
    print(f"  {name:20s}: {val.item():.6f}  (scalar, finite) PASS")

# --- Zero-loss sanity: identical inputs ---
print("\n--- Zero-Loss Sanity (pred == gt) ---")
same = torch.rand(B, C, H, W)
for name, loss_fn in tests[:3]:  # Charbonnier, Edge, FFT should be ~0
    val = loss_fn(same, same)
    print(f"  {name:20s}: {val.item():.8f}")
    assert val.item() < 0.01, f"{name}: loss too high for identical inputs"
print("  PASS: reconstruction losses near zero for identical inputs")

# --- Edge loss on blurred prediction against sharp GT ---
print("\n--- Edge Loss: Blurred Prediction vs Sharp GT ---")
edge_loss_fn = EdgeLoss()
# Create sharp step edge GT
sharp_gt = torch.zeros(B, C, H, W)
sharp_gt[:, :, :, W // 2:] = 1.0  # sharp vertical step edge

# Create blurred prediction using avg_pool blur filter
blurred_pred = F.avg_pool2d(sharp_gt, kernel_size=9, stride=1, padding=4)

edge_loss_sharp = edge_loss_fn(sharp_gt, sharp_gt)
edge_loss_blurred = edge_loss_fn(blurred_pred, sharp_gt)
print(f"  Edge Loss (sharp GT vs sharp GT)   : {edge_loss_sharp.item():.8f}")
print(f"  Edge Loss (blurred pred vs sharp GT): {edge_loss_blurred.item():.8f}")
assert edge_loss_blurred.item() > 0.05, "Edge loss must be nonzero and substantial for blurred edges"
assert edge_loss_blurred.item() > edge_loss_sharp.item() * 10, "Blurred edge loss must exceed sharp edge baseline"
print("  PASS: Edge loss is strictly non-zero and strongly penalizes blurred edges")

# --- FFT loss on frequency-mismatched prediction ---
print("\n--- FFT Loss: Frequency-Mismatched Prediction vs High-Freq GT ---")
fft_loss_fn = FFTLoss()
# Create high-frequency patterned GT
x_coords = torch.linspace(0, 32 * 3.14159, W)
y_coords = torch.linspace(0, 32 * 3.14159, H)
grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
high_freq_gt = (torch.sin(grid_x) * torch.cos(grid_y)).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)

# Frequency-mismatched prediction (heavily smoothed/low-pass filtered)
low_freq_pred = F.avg_pool2d(high_freq_gt, kernel_size=15, stride=1, padding=7)

fft_match = fft_loss_fn(high_freq_gt, high_freq_gt)
fft_mismatch = fft_loss_fn(low_freq_pred, high_freq_gt)
print(f"  FFT Loss (matched high-freq)   : {fft_match.item():.8f}")
print(f"  FFT Loss (frequency mismatched): {fft_mismatch.item():.8f}")
assert fft_mismatch.item() > 0.005, "FFT loss must be nonzero for frequency-mismatched predictions"
assert fft_mismatch.item() > fft_match.item() + 1e-4, "Frequency mismatch must yield higher FFT loss"
print("  PASS: FFT loss is strictly non-zero and detects frequency spectrum mismatch")

# --- Range penalty: should penalize out-of-range ---
print("\n--- Range Penalty Specifics ---")
rp = RangePenaltyLoss()
in_range = torch.rand(B, C, H, W)  # all in [0,1]
out_range = torch.rand(B, C, H, W) * 3 - 1  # some in [-1, 2]
rp_in = rp(in_range, gt)
rp_out = rp(out_range, gt)
print(f"  In-range [0,1]:  {rp_in.item():.8f}")
print(f"  Out-range [-1,2]: {rp_out.item():.8f}")
assert rp_out > rp_in, "Range penalty should be higher for out-of-range"
print("  PASS: out-of-range penalized more")

# --- Composite loss from config & weight verification ---
print("\n--- Composite Loss (config-driven weights) ---")
cfg = load_config("configs/hansr.yaml")
composite = CompositeLoss(cfg)
print(composite)

# Verify updated weights for KLA detail restoration
assert composite.weights["charbonnier"] == 1.0, f"Expected charb weight 1.0, got {composite.weights.get('charbonnier')}"
assert composite.weights["edge"] == 0.25, f"Expected edge weight 0.25, got {composite.weights.get('edge')}"
assert composite.weights["fft"] == 0.15, f"Expected fft weight 0.15, got {composite.weights.get('fft')}"
assert composite.weights["range_penalty"] == 0.01, f"Expected range_penalty weight 0.01, got {composite.weights.get('range_penalty')}"
print("  PASS: all loss weights correctly set (charb=1.0, edge=0.25, fft=0.15, range=0.01)")

pred_g = pred.clone().requires_grad_(True)
total, breakdown = composite(pred_g, gt)
print(f"\n  Total loss: {total.item():.6f}")
for k, v in breakdown.items():
    if k != "total":
        w = composite.weights.get(k, 0)
        print(f"  {k:20s}: raw={v:.6f}  weight={w}  contrib={v*w:.6f}")
assert torch.isfinite(total), "Total loss not finite"
assert total.requires_grad, "Total loss must have grad"
print("  PASS: composite loss is finite and differentiable")

# --- Gradient flow through composite ---
print("\n--- Gradient Flow ---")
p = torch.randn(B, C, H, W, requires_grad=True)
g = torch.randn(B, C, H, W)
t, _ = composite(p, g)
t.backward()
assert p.grad is not None, "No gradient on pred"
assert torch.isfinite(p.grad).all(), "Non-finite gradients"
print("  PASS: gradients flow through all active terms")

# --- Ablation: disable one term ---
print("\n--- Ablation: disable edge loss ---")
cfg_no_edge = load_config("configs/hansr.yaml")
cfg_no_edge["loss"]["edge"]["weight"] = 0.0
comp_no_edge = CompositeLoss(cfg_no_edge)
assert "edge" not in comp_no_edge.terms, "Edge should be removed when weight=0"
print(f"  Active terms: {list(comp_no_edge.terms.keys())}")
print("  PASS: terms with weight=0 are excluded")

print("\n" + "=" * 60)
print("=== PHASE 3 COMPLETE — ALL LOSS CHECKS PASSED ===")
print("=" * 60)
