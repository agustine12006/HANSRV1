"""Phase 3 verification — loss function tests."""
import sys
sys.path.insert(0, ".")
import torch
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

# --- FFT anti-hallucination: high-freq noise should increase loss ---
print("\n--- FFT Anti-Hallucination Check ---")
fft_loss = FFTLoss()
clean = torch.rand(B, C, H, W) * 0.5 + 0.25
noisy = clean + torch.randn_like(clean) * 0.3  # added high-freq noise
fft_clean = fft_loss(clean, clean)
fft_noisy = fft_loss(noisy, clean)
print(f"  Clean vs clean:  {fft_clean.item():.8f}")
print(f"  Noisy vs clean:  {fft_noisy.item():.8f}")
assert fft_noisy > fft_clean, "FFT loss should detect added high-freq energy"
print("  PASS: FFT detects unsupported high-frequency energy")

# --- Composite loss from config ---
print("\n--- Composite Loss (config-driven) ---")
cfg = load_config("configs/hansr.yaml")
composite = CompositeLoss(cfg)
print(composite)
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
print("  PASS: gradients flow through all five terms")

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
