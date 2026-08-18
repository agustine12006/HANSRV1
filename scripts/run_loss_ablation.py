"""
Controlled Loss Balance Ablation Study for KLA Fine-Detail Restoration.

Evaluates:
  A: Charbonnier only (1.0, edge=0, fft=0, range=0.01)
  B: Charbonnier + Edge (1.0, edge=0.20, fft=0, range=0.01)
  C: Charbonnier + Edge + FFT balanced (1.0, edge=0.20, fft=0.10, range=0.01)
  D: Current full loss (1.0, edge=0.25, fft=0.15, range=0.01)

Reports:
  - Val PSNR
  - Val SSIM
  - Edge Error (L_edge against GT)
  - FFT Frequency Error (L_fft against GT)
  - Ringing/Halo metrics
"""

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, ".")

from hansr.model import build_model, count_parameters
from hansr.losses import CompositeLoss, EdgeLoss, FFTLoss
from hansr.metrics import compute_psnr, compute_ssim
from hansr.utils import load_config, set_seed, get_device
from hansr.degradation import degrade_image


class SyntheticBenchmarkDataset(Dataset):
    """Generates synthetic high-resolution SEM/wafer patterns with thin lines and contact holes."""
    def __init__(self, num_samples=32, scale=2, seed=42):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.samples = []
        for i in range(num_samples):
            # Create synthetic wafer-like image: background + lines + contact holes + steps
            img = np.zeros((256, 256), dtype=np.float32) + 0.2
            
            # Step regions
            img[30:100, 30:220] = 0.8
            img[120:200, 50:180] = 0.5
            
            # Thin periodic grating / lines (1-2 px wide)
            for x in range(40, 210, 8):
                img[40:90, x:x+2] = 0.1
                img[130:190, x:x+3] = 0.9

            # Contact holes (small circles/dots)
            for cy in range(210, 245, 12):
                for cx in range(40, 220, 16):
                    img[cy-2:cy+3, cx-2:cx+3] = 0.95
                    img[cy-1:cy+2, cx-1:cx+2] = 0.05

            gt = torch.from_numpy(img).unsqueeze(0)  # (1, 256, 256)
            
            # Degrade
            deg_cfg = {
                "gaussian": {"sigma_range": [0.03, 0.05]},
                "speckle": {"sigma_range": [0.1, 0.2]},
                "downsample": {"factor": scale},
            }
            deg, _ = degrade_image(gt, deg_cfg, case=("speckle", "gaussian", "resolution"))
            self.samples.append({"gt": gt, "degraded": deg})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def evaluate_fidelity(model, val_loader, device):
    model.eval()
    edge_fn = EdgeLoss().to(device)
    fft_fn = FFTLoss().to(device)

    total_psnr = 0.0
    total_ssim = 0.0
    total_edge_err = 0.0
    total_fft_err = 0.0
    max_overshoot = 0.0

    with torch.no_grad():
        for batch in val_loader:
            deg = batch["degraded"].to(device)
            gt = batch["gt"].to(device)
            pred = model(deg)
            pred_clamped = pred.clamp(0.0, 1.0)

            total_psnr += compute_psnr(pred_clamped, gt)
            total_ssim += compute_ssim(pred_clamped, gt)
            total_edge_err += edge_fn(pred, gt).item()
            total_fft_err += fft_fn(pred, gt).item()

            # Measure overshoot/undershoot outside [0, 1] before clamping
            overshoot = max(0.0, (pred - 1.0).max().item(), (-pred).max().item())
            max_overshoot = max(max_overshoot, overshoot)

    n = len(val_loader)
    return {
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
        "edge_error": total_edge_err / n,
        "fft_error": total_fft_err / n,
        "max_overshoot": max_overshoot,
    }


def run_ablation():
    set_seed(42)
    device = get_device()
    print(f"Running Loss Balance Ablation on {device}...")

    train_ds = SyntheticBenchmarkDataset(num_samples=48, seed=100)
    val_ds = SyntheticBenchmarkDataset(num_samples=16, seed=200)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    base_cfg = load_config("configs/hansr.yaml")
    base_cfg["model"]["width"] = 16  # Fast test width for rapid controlled comparison
    base_cfg["model"]["num_blocks"] = [1, 2, 2, 4]
    base_cfg["model"]["middle_blocks"] = 2

    configs = {
        "A (Charbonnier Only)": {
            "charbonnier": {"weight": 1.0, "eps": 1e-3},
            "edge": {"weight": 0.0},
            "fft": {"weight": 0.0},
            "range_penalty": {"weight": 0.01},
        },
        "B (Charbonnier + Edge 0.20)": {
            "charbonnier": {"weight": 1.0, "eps": 1e-3},
            "edge": {"weight": 0.20},
            "fft": {"weight": 0.0},
            "range_penalty": {"weight": 0.01},
        },
        "C (Balanced: Edge 0.20 + FFT 0.10)": {
            "charbonnier": {"weight": 1.0, "eps": 1e-3},
            "edge": {"weight": 0.20},
            "fft": {"weight": 0.10},
            "range_penalty": {"weight": 0.01},
        },
        "D (Current: Edge 0.25 + FFT 0.15)": {
            "charbonnier": {"weight": 1.0, "eps": 1e-3},
            "edge": {"weight": 0.25},
            "fft": {"weight": 0.15},
            "range_penalty": {"weight": 0.01},
        },
        "E (High-Fidelity: Edge 0.20 + FFT 0.05)": {
            "charbonnier": {"weight": 1.0, "eps": 1e-3},
            "edge": {"weight": 0.20},
            "fft": {"weight": 0.05},
            "range_penalty": {"weight": 0.01},
        },
    }

    num_epochs = 12
    results = {}

    for name, loss_dict in configs.items():
        print(f"\n==================================================")
        print(f"Training Config: {name}")
        print(f"==================================================")
        cfg = copy.deepcopy(base_cfg)
        cfg["loss"] = loss_dict

        set_seed(42)
        model = build_model(cfg).to(device)
        criterion = CompositeLoss(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

        for epoch in range(num_epochs):
            model.train()
            ep_loss = 0.0
            for batch in train_loader:
                deg = batch["degraded"].to(device)
                gt = batch["gt"].to(device)
                optimizer.zero_grad()
                pred = model(deg)
                loss, _ = criterion(pred, gt)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()

        metrics = evaluate_fidelity(model, val_loader, device)
        results[name] = metrics
        print(f"  -> Val PSNR     : {metrics['psnr']:.2f} dB")
        print(f"  -> Val SSIM     : {metrics['ssim']:.4f}")
        print(f"  -> Edge Error   : {metrics['edge_error']:.6f} (lower = sharper/cleaner edges)")
        print(f"  -> FFT Error    : {metrics['fft_error']:.6f} (lower = better frequency match)")
        print(f"  -> Max Overshoot: {metrics['max_overshoot']:.4f}")

    print("\n" + "=" * 80)
    print(f"{'Config':<42} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Edge Err':<10} | {'FFT Err':<10}")
    print("=" * 80)
    for name, res in results.items():
        print(f"{name:<42} | {res['psnr']:<10.2f} | {res['ssim']:<8.4f} | {res['edge_error']:<10.6f} | {res['fft_error']:<10.6f}")
    print("=" * 80)

    # Save results
    with open("results_ablation.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_ablation()
