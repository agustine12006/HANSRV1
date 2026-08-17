"""
HANSR Standalone Inference and Evaluation Script (FR-012, FR-015)

Strict isolation: DOES NOT import any dashboard or Streamlit code (FR-015).

Usage:
    python evaluate.py --weights checkpoints/best.pt --input_dir data/test/degraded --output_dir results/restored
    python evaluate.py --weights checkpoints/best.pt --input_dir data/test/degraded --output_dir results/restored --gt_dir data/test/gt

Requirements satisfied:
    FR-001: Blind restoration of input images
    FR-002: Resolution scaling (128->256 or 256->512)
    FR-003: Single-channel grayscale output
    FR-012: Standalone inference CLI with evaluation output
    FR-015: Observability isolation
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from hansr.model import build_model, count_parameters
from hansr.dataset import discover_images, load_grayscale
from hansr.metrics import compute_psnr, compute_ssim, compute_lpips
from hansr.utils import load_config, get_device, load_checkpoint, ensure_dir, setup_logging

logger = logging.getLogger("hansr")


def save_image_tensor(tensor: torch.Tensor, path: str) -> None:
    """
    Save 1-channel tensor (1, H, W) or (H, W) as grayscale image.
    Clips output to [0, 1] for saving.
    """
    arr = tensor.detach().cpu().squeeze().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Run forward pass on a single input image tensor (1, H, W).
    Returns restored tensor (1, 2H, 2W).
    """
    model.eval()
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)  # (1, 1, H, W)

    input_tensor = input_tensor.to(device)
    output_tensor = model(input_tensor)
    return output_tensor.squeeze(0)  # (1, 2H, 2W)


def main():
    parser = argparse.ArgumentParser(description="HANSR Standalone Evaluation & Inference")
    parser.add_argument("--weights", required=True, help="Path to checkpoint weights file (.pt)")
    parser.add_argument("--input_dir", required=True, help="Path to input degraded image directory")
    parser.add_argument("--output_dir", required=True, help="Path to directory where restored images will be saved")
    parser.add_argument("--gt_dir", default=None, help="Optional: Path to GT images directory for benchmark evaluation")
    parser.add_argument("--config", default=None, help="Optional: Path to YAML config (if not stored in checkpoint)")
    parser.add_argument("--save_metrics", default=None, help="Path to output JSON metrics summary (default: output_dir/metrics.json)")
    args = parser.parse_args()

    setup_logging()

    device = get_device()
    output_dir = ensure_dir(args.output_dir)

    # 1. Load checkpoint and model
    if not os.path.exists(args.weights):
        logger.error(f"Weights file not found: {args.weights}")
        sys.exit(1)

    logger.info(f"Loading checkpoint: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)

    config = checkpoint.get("config")
    if config is None:
        if args.config and os.path.exists(args.config):
            config = load_config(args.config)
        elif os.path.exists("configs/hansr.yaml"):
            config = load_config("configs/hansr.yaml")
        else:
            logger.error("No config found in checkpoint and no --config provided.")
            sys.exit(1)

    model = build_model(config).to(device)
    load_checkpoint(args.weights, model=model, device=device, resume_training=False)
    model.eval()

    params = count_parameters(model)
    logger.info(f"Loaded model ({params['total']:,} parameters)")

    # 2. Discover input images
    input_files = discover_images(args.input_dir)
    logger.info(f"Found {len(input_files)} input images in {args.input_dir}")

    # Discover GT files if provided
    gt_map = {}
    if args.gt_dir:
        gt_files = discover_images(args.gt_dir)
        gt_map = {Path(f).stem: f for f in gt_files}
        logger.info(f"Found {len(gt_files)} GT images in {args.gt_dir}")

    # 3. Process images
    per_image_results = []
    total_time = 0.0

    for input_path in tqdm(input_files, desc="Restoring"):
        stem = Path(input_path).stem
        ext = Path(input_path).suffix
        out_path = os.path.join(output_dir, f"{stem}{ext}")

        input_tensor = load_grayscale(input_path)  # (1, H, W)

        t0 = time.time()
        output_tensor = run_inference(model, input_tensor, device)  # (1, 2H, 2W)
        elapsed = time.time() - t0
        total_time += elapsed

        # Save restored image
        save_image_tensor(output_tensor, out_path)

        # Calculate metrics if GT exists
        image_metrics = {"filename": os.path.basename(input_path), "time_sec": elapsed}

        if stem in gt_map:
            gt_tensor = load_grayscale(gt_map[stem]).to(device)
            if gt_tensor.ndim == 3:
                gt_tensor = gt_tensor.unsqueeze(0)

            pred_tensor = output_tensor.unsqueeze(0).clamp(0.0, 1.0)

            psnr_val = compute_psnr(pred_tensor, gt_tensor)
            ssim_val = compute_ssim(pred_tensor, gt_tensor)
            lpips_val = compute_lpips(pred_tensor, gt_tensor, device=device)

            image_metrics["psnr"] = psnr_val
            image_metrics["ssim"] = ssim_val
            image_metrics["lpips"] = lpips_val

        per_image_results.append(image_metrics)

    avg_time = total_time / max(len(input_files), 1)
    logger.info(f"Restoration finished! Avg inference time: {avg_time*1000:.1f} ms/img")

    # 4. Save and report evaluation summary if GT was provided
    if gt_map and per_image_results:
        psnr_list = [r["psnr"] for r in per_image_results if "psnr" in r]
        ssim_list = [r["ssim"] for r in per_image_results if "ssim" in r]
        lpips_list = [r["lpips"] for r in per_image_results if "lpips" in r]

        summary = {
            "num_images": len(psnr_list),
            "mean_psnr": float(np.mean(psnr_list)) if psnr_list else 0.0,
            "std_psnr": float(np.std(psnr_list)) if psnr_list else 0.0,
            "mean_ssim": float(np.mean(ssim_list)) if ssim_list else 0.0,
            "std_ssim": float(np.std(ssim_list)) if ssim_list else 0.0,
            "mean_lpips": float(np.mean(lpips_list)) if lpips_list else 0.0,
            "std_lpips": float(np.std(lpips_list)) if lpips_list else 0.0,
            "avg_inference_time_sec": avg_time,
            "per_image": per_image_results,
        }

        metrics_json_path = args.save_metrics or os.path.join(output_dir, "metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("=" * 50)
        logger.info("EVALUATION BENCHMARK RESULTS")
        logger.info("=" * 50)
        logger.info(f"Evaluated images: {summary['num_images']}")
        logger.info(f"PSNR : {summary['mean_psnr']:.2f} ± {summary['std_psnr']:.2f} dB")
        logger.info(f"SSIM : {summary['mean_ssim']:.4f} ± {summary['std_ssim']:.4f}")
        logger.info(f"LPIPS: {summary['mean_lpips']:.4f} ± {summary['std_lpips']:.4f}")
        logger.info(f"Saved evaluation metrics to: {metrics_json_path}")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()
