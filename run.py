#!/usr/bin/env python3
"""
HANSR — KLA Submission Inference Runner

Usage:
    python run.py <input-dir> <output-dir>

Requirements:
    - Reads all .npy files in <input-dir>
    - Accepts shapes (1, H, W), (H, W), or (H, W, 1) as float32
    - Restores image with 2x super-resolution using trained HANSR model
    - Outputs (2H, 2W) grayscale float32 .npy arrays clipped to [0, 1]
    - Fully offline, deterministic, CUDA/CPU auto-detection
    - Loads checkpoint from models/best.pt
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

from hansr.model import HANSRNet, build_model


def get_device() -> torch.device:
    """Auto-detect CUDA or fallback to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_checkpoint() -> str:
    """Find packaged model checkpoint in models/ or fallback locations."""
    candidates = [
        "models/best.pt",
        "models/model.pt",
        "checkpoints/best.pt",
        "checkpoints/latest.pt",
        os.path.join(os.path.dirname(__file__), "models", "best.pt"),
        os.path.join(os.path.dirname(__file__), "checkpoints", "best.pt"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # Check any .pt in models/
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if os.path.isdir(models_dir):
        pts = sorted([os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith(".pt")])
        if pts:
            return pts[0]

    raise FileNotFoundError(
        "Checkpoint not found. Please ensure 'best.pt' is present in the 'models/' directory "
        "(e.g., models/best.pt)."
    )


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load HANSR model and weights strictly offline."""
    print(f"[HANSR] Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "config" in checkpoint and checkpoint["config"] is not None:
        model = build_model(checkpoint["config"])
    else:
        # Default baseline HANSR architecture configuration
        model = HANSRNet(
            in_channels=1,
            out_channels=1,
            width=32,
            num_blocks=[2, 4, 4, 8],
            middle_blocks=4,
            dropout_rate=0.0,
            upscale_factor=2,
        )

    # Extract state dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Clean state dict keys (strip 'module.' prefix if trained under DDP)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[key] = v

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def preprocess_array(arr: np.ndarray) -> torch.Tensor:
    """
    Accepts (1, H, W), (H, W), or (H, W, 1) and converts to (1, 1, H, W) float32 tensor.
    Preserves unclipped input values for model processing.
    """
    arr = arr.astype(np.float32)

    if arr.ndim == 2:
        # (H, W) -> (1, 1, H, W)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        # (1, H, W) -> (1, 1, H, W)
        tensor = torch.from_numpy(arr).unsqueeze(0)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        # (H, W, 1) -> (1, 1, H, W)
        tensor = torch.from_numpy(arr[:, :, 0]).unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(
            f"Unsupported input array shape: {arr.shape}. Expected (1, H, W), (H, W), or (H, W, 1)."
        )

    return tensor


def postprocess_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts (1, 1, 2H, 2W) tensor to (2H, 2W) float32 numpy array.
    Ensures finite values and clips to [0.0, 1.0].
    """
    arr = tensor.detach().cpu().squeeze().numpy()

    # Handle 1D or empty edge cases
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])

    # Replace non-finite values safely
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

    # Strictly clip to [0.0, 1.0] float32
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    return arr


@torch.no_grad()
def run_pipeline(input_dir: str, output_dir: str):
    """Main inference loop across all .npy files in input directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    npy_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() == ".npy" and f.is_file()])
    if not npy_files:
        print(f"[HANSR] No .npy files found in input directory: {input_dir}")
        return

    device = get_device()
    print(f"[HANSR] Running on device: {device}")
    print(f"[HANSR] Found {len(npy_files)} .npy files to process.")

    checkpoint_path = find_checkpoint()
    model = load_model(checkpoint_path, device)

    processed_count = 0
    for file_p in npy_files:
        try:
            raw_data = np.load(str(file_p))
        except Exception as e:
            print(f"Error loading {file_p}: {e}", file=sys.stderr)
            continue

        inp_tensor = preprocess_array(raw_data).to(device)
        out_tensor = model(inp_tensor)
        out_array = postprocess_tensor(out_tensor)

        # Output shape validation
        assert out_array.ndim == 2, f"Output array must be 2D grayscale, got {out_array.shape}"
        assert out_array.dtype == np.float32, f"Output array must be float32, got {out_array.dtype}"
        assert np.isfinite(out_array).all(), "Output array contains non-finite values"

        dest_file = output_path / file_p.name
        np.save(str(dest_file), out_array)
        processed_count += 1

    print(f"[HANSR] Successfully restored {processed_count}/{len(npy_files)} files -> {output_dir}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>", file=sys.stderr)
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    run_pipeline(input_dir, output_dir)


if __name__ == "__main__":
    main()
