"""
HANSR Streamlit Dashboard — Experiment & Restoration Observability (FR-015, FR-016)

Interactive web interface for:
  1. Visual inspection: GT vs Degraded vs Restored + Difference Heatmap
  2. Metric comparison: PSNR, SSIM, LPIPS across experiments
  3. Ablation tracking: Compare performance under different loss term configurations
  4. Model & system diagnostic summary

Note: Strict isolation mandate (FR-015). This file is imported ONLY by Streamlit launcher.
Inference engine (evaluate.py) NEVER imports this file or package.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import streamlit as st
from PIL import Image

from hansr.model import build_model, count_parameters
from hansr.dataset import discover_images, load_grayscale
from hansr.metrics import compute_psnr, compute_ssim, compute_lpips
from hansr.utils import load_config, load_checkpoint, get_device


# Set Streamlit page layout & theme
st.set_page_config(
    page_title="HANSR Observability Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_experiment_reports(results_dir: str):
    """Load all JSON experiment reports from results directory."""
    if not os.path.exists(results_dir):
        return []

    reports = []
    for root, _, files in os.walk(results_dir):
        for f in files:
            if f.endswith(".json"):
                path = os.path.join(root, f)
                try:
                    with open(path, "r") as fp:
                        data = json.load(fp)
                        data["_file_path"] = path
                        reports.append(data)
                except Exception:
                    pass
    return reports


def main():
    st.title("🔬 HANSR Semiconductor Image Restoration Dashboard")
    st.markdown(
        "**High-Accuracy Neural Semiconductor Restoration (HANSR)** — "
        "NAFNet-style blind restoration for wafer defect & inspection images."
    )

    tabs = st.tabs(["🖼️ Interactive Image Restoration", "📊 Benchmark & Ablation Explorer", "⚙️ Model & Config Inspector"])

    # =========================================================================
    # TAB 1: Interactive Image Restoration
    # =========================================================================
    with tabs[0]:
        st.header("Interactive Image Restoration & Difference Mapping")

        col1, col2 = st.columns([1, 2])

        with col1:
            ckpt_path = st.text_input(
                "Checkpoint Path (.pt)",
                value="checkpoints/best.pt" if os.path.exists("checkpoints/best.pt") else "checkpoints/latest.pt",
            )
            device_str = st.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])

            uploaded_file = st.file_uploader("Upload Degraded Image (Grayscale)", type=["png", "jpg", "jpeg", "tif", "bmp"])
            gt_file = st.file_uploader("Upload Ground Truth (Optional, for metrics)", type=["png", "jpg", "jpeg", "tif", "bmp"])

        with col2:
            if uploaded_file is not None:
                deg_img = Image.open(uploaded_file).convert("L")
                deg_arr = np.array(deg_img, dtype=np.float32) / 255.0
                deg_tensor = torch.from_numpy(deg_arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

                gt_tensor = None
                if gt_file is not None:
                    gt_img = Image.open(gt_file).convert("L")
                    gt_arr = np.array(gt_img, dtype=np.float32) / 255.0
                    gt_tensor = torch.from_numpy(gt_arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

                if os.path.exists(ckpt_path):
                    try:
                        device = torch.device(device_str)
                        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                        cfg = checkpoint.get("config") or load_config("configs/hansr.yaml")

                        model = build_model(cfg).to(device)
                        load_checkpoint(ckpt_path, model=model, device=device, resume_training=False)
                        model.eval()

                        with torch.no_grad():
                            output_tensor = model(deg_tensor.to(device)).cpu()
                            output_clamped = output_tensor.clamp(0.0, 1.0)

                        out_arr = output_clamped.squeeze().numpy()

                        # Up-sample degraded image for visual alignment if resolutions differ
                        deg_vis = deg_img.resize((out_arr.shape[1], out_arr.shape[0]), Image.BICUBIC)
                        deg_vis_arr = np.array(deg_vis, dtype=np.float32) / 255.0

                        # Calculate error map (absolute difference)
                        error_map = np.abs(out_arr - (np.array(gt_img, dtype=np.float32)/255.0 if gt_file else deg_vis_arr))

                        # Display images side-by-side
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.subheader("Input (Degraded)")
                            st.image(deg_img, use_container_width=True)
                        with c2:
                            st.subheader("Bicubic Upsample")
                            st.image(deg_vis, use_container_width=True)
                        with c3:
                            st.subheader("HANSR Restored")
                            st.image(out_arr, use_container_width=True)
                        with c4:
                            st.subheader("Error Heatmap")
                            st.image(error_map, use_container_width=True, clamp=True)

                        # Metrics display
                        if gt_tensor is not None:
                            st.markdown("### 📈 Quality Metrics")
                            m_col1, m_col2, m_col3 = st.columns(3)
                            psnr_v = compute_psnr(output_clamped, gt_tensor)
                            ssim_v = compute_ssim(output_clamped, gt_tensor)
                            lpips_v = compute_lpips(output_clamped, gt_tensor, device=device)

                            m_col1.metric("PSNR (dB)", f"{psnr_v:.2f}")
                            m_col2.metric("SSIM", f"{ssim_v:.4f}")
                            m_col3.metric("LPIPS", f"{lpips_v:.4f}")

                    except Exception as e:
                        st.error(f"Error executing restoration: {e}")
                else:
                    st.warning(f"Checkpoint not found: {ckpt_path}. Please provide a valid checkpoint file.")

    # =========================================================================
    # TAB 2: Benchmark & Ablation Explorer
    # =========================================================================
    with tabs[1]:
        st.header("Experiment Benchmark & Ablation Explorer")

        results_dir = st.text_input("Results Directory", value="results")
        reports = load_experiment_reports(results_dir)

        if reports:
            st.subheader(f"Found {len(reports)} Experiment Report(s)")

            summary_data = []
            for r in reports:
                row = {
                    "File": os.path.basename(r.get("_file_path", "")),
                    "Name": r.get("experiment_name", r.get("name", "N/A")),
                    "PSNR (dB)": r.get("mean_psnr", r.get("metrics", {}).get("best_psnr", "N/A")),
                    "SSIM": r.get("mean_ssim", r.get("metrics", {}).get("best_ssim", "N/A")),
                    "LPIPS": r.get("mean_lpips", r.get("metrics", {}).get("best_lpips", "N/A")),
                }
                summary_data.append(row)

            st.dataframe(summary_data, use_container_width=True)
        else:
            st.info(f"No experiment JSON reports found in '{results_dir}'. Run `evaluate.py` or `train.py` to generate reports.")

    # =========================================================================
    # TAB 3: Model & Config Inspector
    # =========================================================================
    with tabs[2]:
        st.header("Model & Configuration Inspector")

        config_path = st.text_input("Config File", value="configs/hansr.yaml")
        if os.path.exists(config_path):
            cfg = load_config(config_path)
            model = build_model(cfg)
            params = count_parameters(model)

            st.markdown(f"### Model Parameter Summary")
            p1, p2, p3 = st.columns(3)
            p1.metric("Total Parameters", f"{params['total']:,}")
            p2.metric("Trainable Parameters", f"{params['trainable']:,}")
            p3.metric("Model Size (Float32)", f"{params['total_mb']:.2f} MB")

            st.markdown("### Configuration (configs/hansr.yaml)")
            st.json(cfg)
        else:
            st.error(f"Config file not found: {config_path}")


if __name__ == "__main__":
    main()
