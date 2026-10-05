# Deep Learning-Based Direction of Arrival Estimation

An interactive demonstration of multi-source Direction of Arrival (DOA) estimation using a modified ResNeXt-50 model and a 12-element Thinned Coprime Array (TCA).

> Inference and visualization demo for the honours project “Deep Learning Based Direction of Arrival Estimation.” It reproduces the core inference pipeline and compares it with classical MUSIC on the same simulated observation.

## Overview

Direction of Arrival estimation recovers the incident angles of one or more sources from a sensor array. This demo uses a 12-element TCA and treats DOA as multi-label classification on a 1° grid from **−60° to +60°**. The network takes the real and imaginary parts of the sample covariance matrix and predicts a source probability for each angular bin. MUSIC is computed from the same observation and plotted beside the network output.

## Key Results

Best configuration in the project report: `cov_t32` (covariance input, `T = 32` snapshots).

| Evaluation setting | Accuracy | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|
| Validation set, SNR = 10 dB | **99.80%** | 98.93% | 98.18% | 99.92% |
| Hold-out test set, 200,000 samples, SNR = 10 dB | **98.74%** | **91.67%** | **90.21%** | **99.38%** |

- Training used **1.6 million** simulated samples, about one ninth of the 15 million in the reference study.
- At 10 dB SNR, `cov_t32` precision was **91.67%**, versus **56.88% for MUSIC** on the same 200,000-sample test set.
- In the 2×2 ablation, covariance input was the dominant factor: at `T = 16` it improved recall by about **65 percentage points** over raw snapshot input.
- Optimal threshold for `cov_t32` was about `τ* = 0.45`. F1 moved only **0.01 percentage points** from the default threshold of 0.50.
- The 1° grid is the precision ceiling. At SNR = 10 dB and `T = 32`, CRB RMSE is about **0.060°**, roughly **8.3×** below the grid quantisation.

These figures are from the full project study. This repository is an inference demo, not a training-data release.

## Demo Features

- Real and imaginary heatmaps of the sample covariance matrix.
- Layout of the 12 TCA sensors.
- CNN probabilities over the 121-bin grid.
- MUSIC pseudospectrum on the same grid.
- Ground-truth angles and the detection threshold.
- SNR slider from −5 dB to 25 dB, threshold slider from 0.10 to 0.90.
- Manual source-angle input.
- Presets: three separated sources, two close sources, five sources, low SNR, one source, eight sources.

## Method

### Array

12-element thinned coprime array (`M = 5`, `N = 6`, `P = 12`). Normalised positions:

```text
[0, 5, 10, 15, 20, 25, 6, 12, 36, 42, 48, 54]
```

Steering vector: `a(θ) = exp(jπp sin(θ))`.

### Simulation

Independent complex Gaussian sources and spatially white noise:

```text
X = AS + N
R̂ = XXᴴ / T
```

`Re(R̂)` and `Im(R̂)` are the two input channels.

### Network

`ResNeXt-50 32x4d`, with:

- first convolution changed from 3 channels to 2
- final layer outputting 121 logits
- a sigmoid on each bin
- binary cross-entropy for the multi-label target

### MUSIC

Noise subspace from the eigendecomposition of `R̂`, pseudospectrum on the same −60° to +60° grid. The demo uses the known source count. Source-count estimation is not part of this visualisation.

## Requirements

- Python 3.8+
- PyTorch 2.4+
- torchvision 0.19+
- NumPy 1.26.4
- Matplotlib 3.9.0
- Optional CUDA GPU

The demo runs on CPU. Training for the report used an NVIDIA GPU.

## Installation

```bash
git clone https://github.com/Dashmic/Doa-estimation.git
cd Doa-estimation
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows PowerShell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Use the CPU wheel from pytorch.org if you do not have CUDA 12.6.

## Checkpoint

Expected path:

```text
checkpoints/cov_t32_best.pth
```

The file is about 267 MB, over GitHub’s 100 MB file limit, so it is not in the git history. Host it as a Release asset or with Git LFS, then place it at that path or pass `--checkpoint`.

Without the file, `run_demo.py` exits with `Checkpoint not found`.

## Run

```bash
python run_demo.py
python run_demo.py --cpu
python run_demo.py --angles -30 10 45
python run_demo.py --angles -5 5 --snr 0
python run_demo.py --checkpoint /path/to/cov_t32_best.pth
```

The window regenerates the observation when you change SNR, threshold, or angles and press Run.

## Layout

```text
.
├── run_demo.py
├── demo_interactive.py
├── requirements.txt
├── checkpoints/                 # not tracked; add cov_t32_best.pth locally
└── README.md
```

## Notes

- Each run draws a new noise realisation, so plots differ.
- Far-field, narrowband, uncorrelated sources, spatially white noise.
- Reported metrics come from separately generated train, validation, and test sets. Dataset scripts and HDF5 files are not in this repo.
- Simulated data only. Weaker at low SNR. No measured array, near-field, wideband, mutual coupling, multipath, or coherent-source evaluation.
- The 1° grid limits angular resolution.
- MUSIC here is given the true source count.

## License

No license yet. All rights reserved until one is added.
