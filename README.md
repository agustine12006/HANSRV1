# HANSR — High-Accuracy Neural Semiconductor Restoration

**NAFNet-style image restoration system for semiconductor inspection images.**
Developed for the **KLA SemiCon AI Hackathon**.

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

```bash
# Clone repository
git clone <repository-url>
cd n1

# Install dependencies (PyTorch CPU / CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Verify System & Environment

Run the system check to verify setup, configuration schema, and reproducible seeding:

```bash
python test_phase1.py
```

---

## 🛠️ Data Preparation Workflow

### 1. Dataset Verification Gate (FR-009)
Run the verification gate to check pairing, channel counts (1-channel grayscale), decodability, and dimension relationships before training:

```bash
python scripts/verify_dataset.py --config configs/hansr.yaml
```

### 2. Generate Leakage-Free Splits (FR-010)
Group source images to prevent specimen data leakage across train/val/test splits:

```bash
python scripts/generate_splits.py --image_dir data/all_gt --output splits.json
```

### 3. Generate Offline Synthetic Degradations (Optional)
Generate pre-degraded paired image sets across all 7 degradation cases:

```bash
python scripts/generate_degraded.py --gt_dir data/train/gt --output_dir data/train/degraded --config configs/hansr.yaml
```

---

## 🏋️ Training Pipeline

Train HANSR using the config-driven pipeline:

```bash
# Train from scratch
python train.py --config configs/hansr.yaml

# Resume training from checkpoint
python train.py --config configs/hansr.yaml --resume checkpoints/latest.pt
```

Key features:
- **Automatic Mixed Precision (AMP)** & Gradient Clipping
- **Cosine Annealing Learning Rate Scheduler**
- **TensorBoard Logging:** `tensorboard --logdir runs`
- **Bit-Exact Checkpointing (FR-011):** Saves RNG state, optimizer state, and loss weights

---

## 🔬 Standalone Evaluation & Inference (FR-012)

Run restoration inference and calculate benchmark metrics without launching any UI or dashboard:

```bash
# Batch restoration with PSNR/SSIM/LPIPS evaluation
python evaluate.py \
    --weights checkpoints/best.pt \
    --input_dir data/test/degraded \
    --output_dir results/restored \
    --gt_dir data/test/gt
```

Outputs:
- Restored 2× super-resolved grayscale images in `results/restored/`
- Evaluation benchmark JSON summary in `results/restored/metrics.json`

---

## 📊 Streamlit Observability Dashboard (FR-015, FR-016)

Launch the interactive monitoring dashboard for side-by-side visual inspection, difference heatmaps, and experiment comparison:

```bash
streamlit run dashboard/app.py
```

Features:
- **Interactive Visual Comparison:** GT vs Input vs Restored + Difference Heatmap
- **Benchmark & Ablation Explorer:** Compare metrics across loss configurations
- **Model & Config Inspector:** Inspect model parameters and active YAML hyper-parameters

---

## 🏗️ Project Architecture

```
n1/
├── configs/
│   └── hansr.yaml             # Central YAML hyperparameter configuration
├── hansr/
│   ├── __init__.py
│   ├── model.py               # NAFNet-style UNet + SimpleGate + SCA + PixelShuffle
│   ├── losses.py              # 5-Term Composite Loss (Charbonnier, Edge, FFT, Range, TV)
│   ├── degradation.py         # 7-Case blind synthetic degradation engine
│   ├── dataset.py             # PairedDataset & SyntheticDataset loaders
│   ├── metrics.py             # PSNR, SSIM, LPIPS calculation engine
│   └── utils.py               # Config loading, seeding, checkpointing, logging
├── dashboard/
│   ├── __init__.py
│   └── app.py                 # Streamlit observability dashboard (isolated)
├── scripts/
│   ├── verify_dataset.py      # Dataset verification gate (FR-009)
│   ├── generate_splits.py     # Leakage-free dataset splitting (FR-010)
│   └── generate_degraded.py   # Standalone synthetic degradation generator
├── docs/
│   └── model_card.md          # Comprehensive HANSR Model Card
├── train.py                   # Config-driven training entry point
├── evaluate.py                # Standalone production inference & evaluation CLI
└── requirements.txt           # Dependency requirements
```

---

## 🧪 Comprehensive Verification Suite

Run all verification test scripts sequentially:

```bash
python test_phase1.py  # Environment & Scaffolding
python test_phase2.py  # Model Architecture & Shapes
python test_phase3.py  # Composite Loss Terms & Gradients
python test_phase4.py  # Synthetic Degradation & Dataset
python test_phase5.py  # Training Pipeline & Checkpointing
python test_phase6.py  # Standalone Inference CLI & Isolation
python test_phase7.py  # Dashboard Validity & Isolation Gate
```
