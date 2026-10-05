# DOA-CNN-TCA-ResNeXt

<div align="center">

**Deep Learning Based Direction of Arrival Estimation**  
**基于深度学习的波达方向估计**

PolyU EIE4127 Final Year Project | PyTorch 2.1.0+cu126 | RTX 4090 D

[![GitHub](https://img.shields.io/badge/GitHub-eastshg365--cmd%2FDOA--CNN--TCA--ResNeXt-blue?logo=github)](https://github.com/eastshg365-cmd/DOA-CNN-TCA-ResNeXt)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.6-green)
![Smoke Test](https://img.shields.io/badge/Smoke_Test-9%2F9_PASS-brightgreen)

</div>

---

## Overview

Reproduces **Wang et al., IEEE MLSP 2020** — *"A Unified Approach for Target Direction Finding Based on Convolutional Neural Networks"* — using a modified **ResNeXt-50** on a **Thinned Coprime Array (TCA)**.

### Key Results (SNR = 10 dB)

| Model | Input | T | Val Acc | Precision | Recall | Specificity |
|-------|-------|:-:|:-------:|:---------:|:------:|:-----------:|
| CNN-RAW-T16 | Raw | 16 | 94.56% | ~85% | ~75% | ~98% |
| CNN-RAW-T32 | Raw | 32 | 96.67% | ~92% | ~85% | ~99% |
| CNN-COV-T16 | Cov | 16 | 98.78% | ~94% | ~90% | ~99% |
| **CNN-COV-T32** | **Cov** | **32** | **99.80%** | **97.66%** | **96.79%** | **99.63%** |
| Paper (Wang 2020) | Cov | 32 | 99.69% | 97.78% | 97.65% | 99.84% |

> ✅ **1/9 training data** (1.6M vs 15M samples) — all metrics within **< 1 pp** of the paper.

### Core Contributions

| # | Finding | Key Number |
|---|---------|-----------|
| C1 | Data efficiency — ResNeXt-50 generalises with 11% of paper data | Gap < 1 pp |
| C2 | 2×2 factorial ablation isolates Cov vs Raw & T=16 vs T=32 effects | Cov−Raw = +4.22 pp |
| C3 | Threshold gain inversely proportional to model calibration quality | raw_t16: +11.10 pp F1 |
| C4 | Cov model more fragile than raw under extreme perturbation (counterintuitive) | ε=0.20d: −11.75 pp |
| **C5** | **CRB gap: 8.3× precision headroom beyond discrete 1° grid** | **0.5° vs 0.060°** |

---

## Project Structure

```
DOA-CNN-TCA-ResNeXt/
├── configs/                    # YAML hyperparameters (4 model configs)
│   ├── raw_t16.yaml
│   ├── raw_t32.yaml
│   ├── cov_t16.yaml
│   └── cov_t32.yaml
├── datasets/
│   ├── array_geometry.py       # TCA sensor positions: M=5, N=6 → 12 sensors
│   ├── generate_raw.py         # Raw signal generator (2, 12, T)
│   ├── generate_cov.py         # Covariance matrix generator (2, 12, 12)
│   └── data_loader.py          # PyTorch Dataset + DataLoader
├── models/
│   └── resnext_doa.py          # ResNeXt-50: 2ch input, FC(121) + Sigmoid
├── train/
│   └── trainer.py              # BCELoss + AdamW + early stopping + TensorBoard
├── eval/
│   ├── metrics.py              # Accuracy / Precision / Recall / Specificity
│   ├── compare_classical.py    # MUSIC + ESPRIT benchmark
│   └── visualize.py            # Auto-generate all paper figures (Fig.4–8)
├── extensions/
│   ├── ex1_focal_loss/
│   │   ├── focal_loss.py       # Focal Loss (γ=2, α=0.25)
│   │   └── train_focal.py      # EX1: Focal Loss training
│   ├── ex2_threshold_opt.py    # EX2: Grid search optimal threshold
│   ├── ex3_perturbation.py     # EX3: Array position perturbation robustness
│   └── ex4_crb_analysis.py     # EX4: Cramér-Rao Bound comparison
├── math_interference/
│   ├── math_optimized_final.nb # Mathematica notebook (28-page derivation)
│   └── math_virtual_coarray.wl # Virtual difference co-array analysis
├── results/                    # Auto-generated (not committed to git)
│   ├── checkpoints/            # Model weights (.pth)
│   ├── tables/                 # LaTeX + CSV tables
│   └── figures/                # Publication-ready plots
├── Results_Log/
│   ├── RESEARCH_LOGBOOK_20260328.md    # Full research logbook (Chinese)
│   └── RESEARCH_LOGBOOK_20260328_EN.md # Full research logbook (English)
├── smoke_test/run_smoke_test.py
├── requirements.txt
├── train_all.ps1               # Windows PowerShell one-shot pipeline
└── version.py
```

---

## Deployment Guide

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| Python | 3.10 | 3.10.6 |
| CUDA | 12.1 | 12.6 |
| GPU VRAM | 16 GB | 24 GB (RTX 4090) |
| RAM | 32 GB | 64 GB |
| Disk | 50 GB | 200 GB (full 8M dataset) |

### Step 1 — Clone the Repository

```bash
git clone https://github.com/eastshg365-cmd/DOA-CNN-TCA-ResNeXt.git
cd DOA-CNN-TCA-ResNeXt
```

### Step 2 — Create Python Environment

```bash
# Using conda (recommended)
conda create -n doa python=3.10.6
conda activate doa

# Or using venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### Step 3 — Install PyTorch (CUDA 12.6)

```bash
# CUDA 12.6
pip install torch==2.1.0+cu126 torchvision==0.16.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# CUDA 12.1 (alternative)
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# CPU only (slow, for testing only)
pip install torch torchvision
```

### Step 4 — Install Other Dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Verify Installation (Smoke Test)

```bash
python smoke_test/run_smoke_test.py
# Expected: 9/9 PASS
```

---

## Running the Pipeline

### Option A — Full Pipeline (Windows PowerShell)

```powershell
# Runs all 4 models sequentially (~72h total on RTX 4090)
.\train_all.ps1
```

### Option B — Step-by-Step

#### 1. Generate Data

```bash
# Raw signal datasets (T=16 and T=32)
python datasets/generate_raw.py --T 16 --samples 1600000 --out data/raw_t16_train.h5
python datasets/generate_raw.py --T 16 --samples  200000 --out data/raw_t16_val.h5
python datasets/generate_raw.py --T 16 --samples  200000 --out data/raw_t16_test.h5

python datasets/generate_raw.py --T 32 --samples 1600000 --out data/raw_t32_train.h5
python datasets/generate_raw.py --T 32 --samples  200000 --out data/raw_t32_val.h5
python datasets/generate_raw.py --T 32 --samples  200000 --out data/raw_t32_test.h5

# Covariance matrix datasets
python datasets/generate_cov.py --T 16 --samples 1600000 --out data/cov_t16_train.h5
python datasets/generate_cov.py --T 16 --samples  200000 --out data/cov_t16_val.h5
python datasets/generate_cov.py --T 16 --samples  200000 --out data/cov_t16_test.h5

python datasets/generate_cov.py --T 32 --samples 1600000 --out data/cov_t32_train.h5
python datasets/generate_cov.py --T 32 --samples  200000 --out data/cov_t32_val.h5
python datasets/generate_cov.py --T 32 --samples  200000 --out data/cov_t32_test.h5
```

> ⚠️ Quick test: use `--samples 10000` to verify the pipeline end-to-end.

#### 2. Train Models

```bash
# Train each of the 4 configurations
python train/trainer.py --config configs/raw_t16.yaml   # ~4.5h, early stop @ep18
python train/trainer.py --config configs/raw_t32.yaml   # ~23.5h, 50 epochs
python train/trainer.py --config configs/cov_t16.yaml   # ~23h,   50 epochs
python train/trainer.py --config configs/cov_t32.yaml   # ~21.6h, 50 epochs

# Monitor training
tensorboard --logdir results/logs
```

#### 3. Evaluate — Standard Metrics

```bash
# SNR sweep for each model (0–20 dB)
python eval/metrics.py --config configs/raw_t16.yaml --snr_range 0 20 --step 2
python eval/metrics.py --config configs/raw_t32.yaml --snr_range 0 20 --step 2
python eval/metrics.py --config configs/cov_t16.yaml --snr_range 0 20 --step 2
python eval/metrics.py --config configs/cov_t32.yaml --snr_range 0 20 --step 2

# Classical baselines: MUSIC + ESPRIT
python eval/compare_classical.py --snr_range 0 20 --step 2 --samples 500

# Generate all figures (Fig.4–Fig.8)
python eval/visualize.py --all
```

#### 4. Extension Experiments

```bash
# EX1: Focal Loss (γ=2, α=0.25) — ~24h training
python extensions/ex1_focal_loss/train_focal.py --config configs/cov_t32.yaml

# EX2: Threshold optimisation — ~14 min
python extensions/ex2_threshold_opt.py

# EX3: Array perturbation robustness — ~15 min
python extensions/ex3_perturbation.py --epsilon 0.0 0.02 0.05 0.10 0.20

# EX4: CRB analysis — ~1 min
python extensions/ex4_crb_analysis.py
```

---

## Pre-trained Model Weights

Model weights (`*.pth`) are **not tracked by git** (>500 MB each).  
Download from Google Drive:

> 📁 [Google Drive — Model Weights](https://drive.google.com/drive/folders/YOUR_FOLDER_ID)  
> Place files in: `results/checkpoints/`

| File | Val Acc | Size |
|------|:-------:|:----:|
| `raw_t16_best.pth` | 94.56% | ~85 MB |
| `raw_t32_best.pth` | 96.67% | ~85 MB |
| `cov_t16_best.pth` | 98.78% | ~85 MB |
| `cov_t32_best.pth` | **99.80%** | ~85 MB |

---

## Array Geometry

**Thinned Coprime Array (TCA)** — M=5, N=6 → **12 sensors**, aperture = 54d

```
Sub-array 1 (step M=5d): {0, 5, 10, 15, 20, 25}
Sub-array 2 (step N=6d): {0, 6, 12}               ← first [M/2]+1 multiples of N
Sub-array 3 (offset):    {36, 42, 48, 54}          ← last N multiples of M
Union (12 positions):    {0, 5, 6, 10, 12, 15, 20, 25, 36, 42, 48, 54} × d
```

Verified against paper Fig. 2: gcd(M,N) = gcd(5,6) = 1 ✓, DOF = 89 ✓

---

## Model Architecture

Modified **ResNeXt-50** (32×4d):

| Layer | Output | Details |
|-------|--------|---------|
| Conv1 | 56×56 | 7×7, 64 ch — **2-channel input** (Real/Imag) |
| MaxPool | 28×28 | 3×3, stride 2 |
| Stage 1–4 | → 4×4 | Standard ResNeXt blocks |
| GAP | 1×1 | Global Average Pooling |
| FC | 121 | Linear(2048→121) + **Sigmoid** |

**Loss:** BCEWithLogitsLoss · **Optimizer:** AdamW (lr=1e-3, wd=1e-2) · **Scheduler:** CosineAnnealingLR

---

## Evaluation Metrics

Per-element over all 121 DOA classes (following paper):

| Metric | Formula |
|--------|---------|
| Accuracy | (TP+TN) / (TP+FP+TN+FN) |
| Precision | TP / (TP+FP) |
| Recall | TP / (TP+FN) |
| Specificity | TN / (TN+FP) |

Default threshold: **τ = 0.5** (cov_t32 F1 loss < 0.01 pp vs optimal τ=0.45)

---

## Theoretical Analysis

See `math_interference/math_optimized_final.nb` (Mathematica, 28-page export):

| Section | Content |
|---------|---------|
| §1 | TCA geometry — numerical verification |
| §2 | Difference co-array (DCA) — 89 elements, DOF ≫ 11 |
| §3 | Steering matrix A[12×121] — phase verification |
| §6 | MUSIC eigenvalue decomposition — λ₃/λ₄ ≈ 8× gap |
| §7 | BCE gradient: ∂L/∂z = σ(z) − y (linear residual) |
| §8 | CRB curves — **8.3× gap** at SNR=10 dB, T=32 |

---

## Citation

```bibtex
@inproceedings{wang2020unified,
  title     = {A Unified Approach for Target Direction Finding Based on Convolutional Neural Networks},
  author    = {Wang, Yue and others},
  booktitle = {IEEE International Workshop on Machine Learning for Signal Processing (MLSP)},
  year      = {2020},
  doi       = {10.1109/MLSP49062.2020.9231787}
}
```

---

## References

1. Wang et al., *"A Unified Approach for Target Direction Finding Based on CNNs,"* IEEE MLSP 2020.
2. L. C. Godara, *"Application of Antenna Arrays to Mobile Communications, Part II,"* Proc. IEEE, 1997.
3. Y. Tian et al., *"Vehicle Positioning with DL-Based DOA Estimation of ID Sources,"* IEEE IoT-J, 2022.
4. W. Liu & S. Weiss, *Wideband Beamforming: Concepts and Techniques,* Wiley, 2010.

---

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.10.6 |
| PyTorch | 2.1.0+cu126 |
| CUDA | 12.6 |
| GPU | NVIDIA RTX 4090 D (24 GB) |
| OS | Windows 11 |

See `requirements.txt` for full dependency list.
