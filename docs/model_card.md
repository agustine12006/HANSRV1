# HANSR Model Card
**High-Accuracy Neural Semiconductor Restoration (HANSR)**

---

## 1. Model Details

- **Model Name:** HANSR (High-Accuracy Neural Semiconductor Restoration)
- **Model Version:** v1.1.0 (High-Frequency Detail Enhanced)
- **Model Architecture:** NAFNet-style UNet Backbone with Local High-Frequency Enhancement (LHF-NAFBlock), Detail-Preserving Skip Fusion, and Lightweight Pre-PixelShuffle Sub-Pixel Reconstruction Head
- **Parameters:** ~17,706,628 (17.71M float32 parameters, ~67.55 MB)
- **Framework:** PyTorch 2.1+ / torchvision
- **Task:** Single-Image Restoration & Super-Resolution (2× upscale) for Semiconductor Inspection Images
- **License:** Proprietary / SemiCon AI Hackathon Submission

---

## 2. Intended Use & Application Domain

- **Primary Domain:** Semiconductor manufacturing, wafer inspection, optical & scanning electron microscopy (SEM) defect detection.
- **Intended Task:** Restoring degraded, low-resolution, noisy inspection images back to high-resolution clean ground truth for downstream AI defect classification and manual verification.
- **Input Contract:** Single-channel grayscale image ($1 \times H \times W$), pixel values in $[0, 1]$ (unclipped during degradation processing per FR-004).
- **Output Contract:** Single-channel grayscale image ($1 \times 2H \times 2W$), super-resolved residual output anchored by non-trainable bicubic upsampling.

---

## 3. Architecture & Design Principles

```
Input (1ch, H x W)
  ├── Fixed Bicubic Upsample (Non-trainable) ──────────────────────────────────> (1ch, 2H x 2W)
  └── Learnable Branch:                                                                │
        Stem: Conv2d(1 -> 32)                                                          │
        Encoder: 4 Stages [2, 4, 4, 8] LHF-NAFBlocks                                   │
        Bottleneck: 4 LHF-NAFBlocks                                                    │
        Decoder: 4 Stages [8, 4, 4, 2] LHF-NAFBlocks + DetailSkipFusion Skips          │
        Reconstruction Head: DWConv + PWConv + Conv2d + PixelShuffle(2)                │
                                     │                                                 │
                                Learned Residual (1ch, 2H x 2W) ───────────────────────┼
                                                                                       ▼
                                                                       Output = Bicubic + Residual
```

- **Nonlinear Activation Replacement:** Uses **SimpleGate** ($x_1 \odot x_2$) to eliminate standard GELU/ReLU activations and reduce computational complexity.
- **Channel Attention:** Uses **Simplified Channel Attention (SCA)** to recalibrate feature map channels without multi-layer perceptrons or sigmoid gates.
- **Bicubic Anchor (FR-005):** Predicts only the high-frequency residual difference relative to a fixed bicubic base, preventing structural collapse and accelerating convergence.

---

## 4. Training & Degradation Pipeline

### 4.1 Blind Degradation Pipeline (7-Case Combination)
HANSR trains blindly against 7 non-empty combinations of three degradation operations:
1. **Multiplicative Speckle Noise:** $Y = X + X \cdot N, \quad N \sim \mathcal{N}(0, \sigma_s^2)$ ($\sigma_s \in [0.05, 0.30]$)
2. **Additive Gaussian Noise:** $Y = X + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma_g^2)$ ($\sigma_g \in [0.01, 0.08]$)
3. **Bicubic 2× Downsampling**

### 4.2 Five-Term Composite Loss Function (FR-007)
$$\mathcal{L}_{\text{composite}} = \lambda_{\text{charb}} \mathcal{L}_{\text{charb}} + \lambda_{\text{edge}} \mathcal{L}_{\text{edge}} + \lambda_{\text{fft}} \mathcal{L}_{\text{fft}} + \lambda_{\text{range}} \mathcal{L}_{\text{range}} + \lambda_{\text{tv}} \mathcal{L}_{\text{tv}}$$

- **Charbonnier Loss ($\lambda = 1.0$):** Differentiable variant of $L_1$ loss for robust pixel-level reconstruction.
- **Sobel Edge Loss ($\lambda = 0.1$):** Charbonnier loss on Sobel gradient magnitude maps to enforce sharp edge structure.
- **FFT Magnitude Loss ($\lambda = 0.05$):** $L_1$ penalty on 2D Fourier frequency spectrum to prevent high-frequency hallucinations and ringing (FR-006).
- **Soft Range Penalty ($\lambda = 0.01$):** Gentle quadratic penalty discouraging predictions outside $[0, 1]$.
- **Total Variation Loss ($\lambda = 0.001$):** Spatial gradient smoothness regularization to suppress high-frequency artifacts.

---

## 5. Evaluation Metrics

- **Peak Signal-to-Noise Ratio (PSNR):** Quantitative pixel fidelity in decibels (dB). Higher is better.
- **Structural Similarity Index (SSIM):** Structural preservation metric in $[0, 1]$. Higher is better.
- **Learned Perceptual Image Patch Similarity (LPIPS):** Perceptual feature distance via AlexNet backbone. Lower is better.

---

## 6. Observability & Isolation Policy (FR-015)

The production inference CLI (`evaluate.py`) is strictly isolated from monitoring code. It imports zero Streamlit or UI packages to guarantee zero latency overhead, zero UI dependency conflicts, and high stability during batch benchmark execution.
