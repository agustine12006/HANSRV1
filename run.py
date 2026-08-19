#!/usr/bin/env python3
"""
HANSR — KLA Submission Inference Runner & End-to-End Benchmark

Usage:
    python run.py <input-dir> <output-dir>
    python run.py <input-dir> <output-dir> --batch-size 8

Requirements:
    - Reads all .npy files in <input-dir>
    - Accepts shapes (1, H, W), (H, W), or (H, W, 1) as float32
    - Restores image with 2x super-resolution using trained HANSR model
    - Outputs (2H, 2W) grayscale float32 .npy arrays clipped to [0, 1]
    - Fully offline, deterministic, CUDA/CPU auto-detection
    - Loads checkpoint from models/best.pt
    - Efficient batched execution
    - Measures and reports true end-to-end inference benchmark:
      (Load -> Preprocess -> Batch -> Model Inference -> Postprocess -> Write to Disk)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from hansr.model import HANSRNet, build_model


def get_device(device_override: Optional[str] = None) -> torch.device:
    """Auto-detect CUDA or fallback to CPU, with explicit override support."""
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_name(device: torch.device) -> str:
    """Return human-readable device/GPU description."""
    if device.type == "cuda" and torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(device)} (CUDA)"
    return "CPU"


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
    Accepts (1, H, W), (H, W), or (H, W, 1) and converts to (1, H, W) float32 tensor.
    Preserves unclipped input values for model processing.
    """
    arr = arr.astype(np.float32)

    if arr.ndim == 2:
        # (H, W) -> (1, H, W)
        tensor = torch.from_numpy(arr).unsqueeze(0)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        # (1, H, W) -> (1, H, W)
        tensor = torch.from_numpy(arr)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        # (H, W, 1) -> (1, H, W)
        tensor = torch.from_numpy(arr[:, :, 0]).unsqueeze(0)
    else:
        raise ValueError(
            f"Unsupported input array shape: {arr.shape}. Expected (1, H, W), (H, W), or (H, W, 1)."
        )

    return tensor


def postprocess_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts (1, 2H, 2W) or (2H, 2W) tensor to (2H, 2W) float32 numpy array.
    Ensures finite values and strictly clips to [0.0, 1.0].
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
def run_pipeline(
    input_dir: str,
    output_dir: str,
    batch_size: int = 8,
    device_override: Optional[str] = None,
):
    """
    Main inference loop across all .npy files in input directory.
    Executes true end-to-end benchmark timing covering:
      1. Loading input .npy images from disk
      2. Preprocessing
      3. Batching
      4. Model inference forward pass
      5. Postprocessing
      6. Writing restored .npy files to disk
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    npy_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() == ".npy" and f.is_file()])
    num_files = len(npy_files)
    if num_files == 0:
        print(f"[HANSR] No .npy files found in input directory: {input_dir}")
        return

    device = get_device(device_override)
    device_desc = get_device_name(device)
    print(f"[HANSR] Running on device: {device_desc}")
    print(f"[HANSR] Found {num_files} .npy files to process (batch_size={batch_size}).")

    checkpoint_path = find_checkpoint()
    model = load_model(checkpoint_path, device)

    # =========================================================================
    # True End-to-End Benchmark Timer
    # Wraps: Disk Read -> Preprocess -> Batch -> Model -> Postprocess -> Disk Write
    # =========================================================================
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t_start = time.perf_counter()
    processed_count = 0

    # Process in batches
    for i in range(0, num_files, batch_size):
        batch_paths = npy_files[i:i + batch_size]

        # 1 & 2. Disk Load + Preprocess
        batch_tensors = []
        valid_paths = []
        for file_p in batch_paths:
            try:
                raw_data = np.load(str(file_p))
                t = preprocess_array(raw_data)
                batch_tensors.append(t)
                valid_paths.append(file_p)
            except Exception as e:
                print(f"Error loading {file_p}: {e}", file=sys.stderr)

        if not batch_tensors:
            continue

        # Check if all images in the batch share identical spatial dimensions for batching
        shapes = [t.shape for t in batch_tensors]
        all_same_shape = all(s == shapes[0] for s in shapes)

        if all_same_shape:
            # 3. Batching
            batch_inp = torch.stack(batch_tensors, dim=0).to(device)  # (B, 1, H, W)
            # 4. Model Inference
            batch_out = model(batch_inp)                               # (B, 1, 2H, 2W)
            # 5 & 6. Postprocess + Disk Write
            for out_t, file_p in zip(batch_out, valid_paths):
                out_arr = postprocess_tensor(out_t)

                assert out_arr.ndim == 2, f"Output array must be 2D grayscale, got {out_arr.shape}"
                assert out_arr.dtype == np.float32, f"Output array must be float32, got {out_arr.dtype}"
                assert np.isfinite(out_arr).all(), "Output array contains non-finite values"

                dest_file = output_path / file_p.name
                np.save(str(dest_file), out_arr)
                processed_count += 1
        else:
            # Fallback for mixed resolutions in the same batch
            for t, file_p in zip(batch_tensors, valid_paths):
                inp = t.unsqueeze(0).to(device)  # (1, 1, H, W)
                out_t = model(inp)               # (1, 1, 2H, 2W)
                out_arr = postprocess_tensor(out_t)

                assert out_arr.ndim == 2, f"Output array must be 2D grayscale, got {out_arr.shape}"
                assert out_arr.dtype == np.float32, f"Output array must be float32, got {out_arr.dtype}"
                assert np.isfinite(out_arr).all(), "Output array contains non-finite values"

                dest_file = output_path / file_p.name
                np.save(str(dest_file), out_arr)
                processed_count += 1

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t_end = time.perf_counter()
    total_elapsed = t_end - t_start

    # Metrics calculation
    throughput = processed_count / total_elapsed if total_elapsed > 0 else 0.0
    avg_latency_ms = (total_elapsed / processed_count * 1000.0) if processed_count > 0 else 0.0

    print(f"[HANSR] Successfully restored {processed_count}/{num_files} files -> {output_dir}")
    print("=" * 60)
    print("HANSR END-TO-END INFERENCE BENCHMARK REPORT")
    print("=" * 60)
    print(f"  Number of Images      : {processed_count}")
    print(f"  Batch Size            : {batch_size}")
    print(f"  GPU / Device          : {device_desc}")
    print(f"  Total End-to-End Time : {total_elapsed:.4f} seconds")
    print(f"  Throughput            : {throughput:.2f} images/second")
    print(f"  Average Latency       : {avg_latency_ms:.2f} ms/image")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="HANSR KLA Submission Inference Runner & End-to-End Benchmark"
    )
    parser.add_argument("input_dir", help="Directory containing input .npy files")
    parser.add_argument("output_dir", help="Directory where restored .npy files will be saved")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override ('cuda', 'cpu', 'cuda:0', etc.)",
    )
    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
