# HANSR — High-Accuracy Neural Semiconductor Restoration

**NAFNet-style image restoration system for semiconductor inspection images.**
Developed for the **KLA SemiCon AI Hackathon**.

---

## 🏆 KLA Submission Execution

To run restoration inference and end-to-end benchmarking on the test set, execute:

```bash
# Standard submission execution (default batch size 8)
python run.py <input-dir> <output-dir>

# Optional: customize batch size or device
python run.py <input-dir> <output-dir> --batch-size 8 --device cuda
```

### Specifications:
- **Input Format**: Reads all `.npy` files inside `<input-dir>`. Supports shapes `(1, H, W)`, `(H, W)`, or `(H, W, 1)` in `float32`.
- **Output Format**: Grayscale 2D `.npy` NumPy array of shape `(2H, 2W)`, dtype `float32`. All values are finite and clipped to `[0.0, 1.0]`. Filenames match the input `.npy` exactly.
- **Resolution**: Fixed 2× super-resolution upscaling (e.g., $128 \times 128 \to 256 \times 256$).
- **Model Checkpoint**: Automatically loads the packaged model from `models/best.pt`.
- **Offline & Autonomous**: Fully offline execution with zero network calls, zero dynamic weight downloads, and zero user interaction required.
- **Compute Acceleration**: Automatically uses CUDA GPU if available, falling back seamlessly to CPU.
- **End-to-End Benchmark**: Automatically reports true end-to-end timing covering disk load, preprocessing, batching, model inference, postprocessing, and writing outputs to disk.

### Submission Package Structure:
```
team_name/
├── run.py
├── requirements.txt
├── README.md
├── hansr/
│   ├── __init__.py
│   ├── model.py
│   ├── dataset.py
│   ├── degradation.py
│   ├── losses.py
│   ├── metrics.py
│   └── utils.py
├── configs/
│   └── hansr.yaml
└── models/
    └── best.pt
```

---

## 🚀 Quick Start & Verification

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Submission Contract

Run the automated KLA submission verification test to ensure compliance with input/output format, resolution, data types, and directory contracts:

```bash
python test_submission.py
```

### 3. Verify Master Verification Test Suite

Run the full Phase 1 to Phase 7 verification suite:

```bash
python test_all_phases.py
```

---

## 🛠️ Data Preparation & Split Generation

### 1. Dataset Verification Gate (FR-009)
Run the dataset gate to verify pairing, channel counts (1-channel grayscale), decodability, and dimension relationships before training:

```bash
python scripts/verify_dataset.py --config configs/hansr.yaml
```

### 2. Generate Leakage-Free 90/10 Splits (FR-010)
Generate reproducible 90/10 train/val splits for the 3,200 KLA paired samples using seed 42 (Train = 2,880 pairs, Val = 320 pairs):

```bash
python scripts/generate_splits.py --image_dir /kaggle/input/datasets/jhansiranimajhi/kla-dataset/train/GT --output splits.json --train_ratio 0.9 --val_ratio 0.1 --test_ratio 0.0 --seed 42
```

---

## 🏋️ Training Pipeline & Reproduction

Train HANSR using the config-driven pipeline:

```bash
# Train from scratch (auto-splits 90/10 train/val if validation directory is not separated)
python train.py --config configs/hansr.yaml

# Resume training from checkpoint (restores model, optimizer, scheduler, and RNG state)
python train.py --config configs/hansr.yaml --resume checkpoints/latest.pt
```

### Reproduction Settings (`configs/hansr.yaml`):
- **Epochs**: 200
- **Batch Size**: 8
- **Patch Size**: 128
- **Learning Rate**: `0.0001` (`1.0e-4`) (stable learning rate preventing NaN divergence)
- **Optimizer**: AdamW (`weight_decay: 0.0`)
- **Scheduler**: Cosine Annealing (`min_lr: 1.0e-6`, `warmup_epochs: 5`)
- **Gradient Clipping**: `1.0`
- **AMP**: `false` (disabled for numerical stability)
- **Seed**: 42

### 5-Term Composite Loss Weights:
- **Charbonnier**: `1.0` (robust pixel-level reconstruction)
- **Edge (Sobel / Multiscale)**: `0.40` (strengthened for line edge roughness and sharp boundary restoration)
- **FFT Magnitude**: `0.25` (strengthened for high-frequency structure recovery and anti-hallucination)
- **Range Penalty**: `0.01` (soft $[0, 1]$ output range constraint)
- **Total Variation**: `0.0` (disabled to prevent over-smoothing)

---

## 🔬 Standalone Evaluation CLI (FR-012)

Run restoration inference and calculate benchmark metrics (PSNR, SSIM, LPIPS) without launching any UI:

```bash
python evaluate.py \
    --weights checkpoints/best.pt \
    --input_dir data/test/degraded \
    --output_dir results/restored \
    --gt_dir data/test/gt
```

Outputs:
- Restored 2× super-resolved grayscale images in `results/restored/`
- Benchmark JSON summary in `results/restored/metrics.json`

---

## 📦 Model Checkpoints & Git LFS

The final trained model weights (`models/best.pt`, ~203 MB) are tracked via Git LFS:
- `.gitattributes` tracks `models/*.pt` and `models/*.pth`.
- Training checkpoint scratch files in `checkpoints/` are git-ignored.

---

## 🏗️ Project Architecture

```
HANSRV1/
├── run.py                     # KLA submission inference CLI & end-to-end benchmark
├── test_submission.py         # KLA submission contract verification script
├── test_all_phases.py         # Master verification test suite (Phases 1-7)
├── train.py                   # Config-driven training entry point with validation metrics
├── evaluate.py                # Standalone evaluation & inference CLI (PSNR, SSIM, LPIPS)
├── requirements.txt           # Dependency requirements
├── README.md                  # Instructions and documentation
├── configs/
│   └── hansr.yaml             # Central YAML hyperparameter configuration
├── hansr/
│   ├── __init__.py
│   ├── model.py               # NAFNet-style UNet (17.7M parameters)
│   ├── losses.py              # 5-Term Composite Loss (Charbonnier, Edge, FFT, Range, TV)
│   ├── degradation.py         # 7-Case blind synthetic degradation engine
│   ├── dataset.py             # PairedDataset (90/10 split) & SyntheticDataset loaders
│   ├── metrics.py             # PSNR, SSIM, LPIPS calculation engine
│   └── utils.py               # Config loading, seeding, checkpointing, ByteTensor RNG restoration
├── dashboard/
│   ├── __init__.py
│   └── app.py                 # Streamlit observability dashboard (isolated)
├── scripts/
│   ├── verify_dataset.py      # Dataset verification gate (FR-009)
│   ├── generate_splits.py     # Leakage-free dataset splitting (FR-010)
│   └── generate_degraded.py   # Standalone synthetic degradation generator
└── models/
    └── .gitkeep               # Directory for final best.pt checkpoint
```
