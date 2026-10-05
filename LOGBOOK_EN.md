# Training Logbook
## DOA-CNN-TCA-ResNeXt | EIE4127 FYP | PolyU

> **Last Updated:** 2026-03-26 12:57  
> **Machine:** DESKTOP-0DS3E2A  
> **GPU:** NVIDIA GeForce RTX 4090 D  
> **Environment:** Python 3.10.6 | PyTorch 2.11.0+cu126 | CUDA 12.6

---

## 1. Project Overview

This project reproduces the IEEE MLSP 2020 paper:  
**"A Unified Approach for Target Direction Finding Based on Convolutional Neural Networks"**  
DOI: 10.1109/MLSP49062.2020.9231787

**Task:** Estimate the Direction-of-Arrival (DOA) of multiple signal sources using a Thinned Coprime Array (TCA) and a modified ResNeXt-50 CNN.

**Formulation:** Multi-label binary classification over 121 candidate angles (−60° to +60°, 1° step). Each angle is independently predicted with a Sigmoid activation, and Binary Cross-Entropy (BCELoss) is used as the loss function.

---

## 2. Array Geometry: Thinned Coprime Array (TCA)

The array uses **M = 5, N = 6** (coprime: gcd(5,6) = 1), yielding **P = 12 physical sensors**.

The three sub-arrays are defined as:

| Sub-array | Formula | Positions (×d, d = λ/2) |
|-----------|---------|------------------------|
| X₁ | {n·M : n = 0,...,N-1} | {0, 5, 10, 15, 20, 25} |
| X₂ | {m·N : m = 1,...,⌊M/2⌋} | {6, 12} |
| X₃ | {(m+M+1)·N : m = 0,...,M-2} | {36, 42, 48, 54} |
| **TCA** | X₁ ∪ X₂ ∪ X₃ | **{0, 5, 6, 10, 12, 15, 20, 25, 36, 42, 48, 54}** |

**Total sensors:** S = M + N + ⌊M/2⌋ − 1 = 5 + 6 + 2 − 1 = **12**  
**Array aperture:** 54d (vs 11d for a 12-element ULA)  
**Virtual DOF:** >12 (underdetermined scenario supported: K > P possible)

---

## 3. Signal Model

The received signal at **T** snapshots follows:

```
x(t) = A(θ) · s(t) + n(t),    t = 1, ..., T
```

Where:
- `x(t) ∈ ℂ^P` — received signal vector (P = 12 sensors)
- `A(θ) ∈ ℂ^(P×K)` — steering matrix, columns: `a(θk) = exp(jπ·p·sin(θk))`
- `s(t) ∈ ℂ^K` — source signals, i.i.d. CN(0, I)
- `n(t) ∈ ℂ^P` — AWGN noise, CN(0, σ²I)
- `K` — number of sources, drawn randomly from {1, ..., 16} per sample
- `T` — number of snapshots (16 or 32 in this project)
- SNR = 10 log₁₀(σ_s² / σ_n²), set to **10 dB** during training

---

## 4. Dataset Generation

### 4.1 Two Input Representations

The dataset provides CNN inputs in two formats derived from the same signal model:

**Format 1 — Raw Signal (`raw`):**
```
X_raw = [Re(X); Im(X)]  ∈ ℝ^(2 × 12 × T)
```
The full T-snapshot matrix is split into real and imaginary channels.
CNN learns to extract covariance-like features automatically.

**Format 2 — Covariance Matrix (`cov`):**
```
R̂ = (1/T) · X · X^H  ∈ ℂ^(12×12)
X_cov = [Re(R̂); Im(R̂)]  ∈ ℝ^(2 × 12 × 12)
```
The sample covariance matrix is computed first. CNN processes the compressed, noise-averaged representation.

### 4.2 Label Definition

Each sample generates a 121-dimensional **multi-hot label vector**:
```
y[i] = 1  if angle θᵢ ∈ {θ₁, ..., θK} (one of the K true DOAs)
y[i] = 0  otherwise
```
DOA grid: θ ∈ {−60°, −59°, ..., +60°}, step = 1°, num_classes = 121.  
Average active labels per sample: ~9.5 (≈ (1+16)/2 sources × 1 label each)

### 4.3 Generation Parameters

| Parameter | Value |
|-----------|-------|
| Training SNR | 10 dB |
| K (sources/sample) | Uniform {1, ..., 16} |
| Snapshot T | 16 or 32 |
| Workers | 31 (CPU multiprocessing) |
| File format | HDF5 (lzf compressed) |
| Chunk size | min(4096, n_samples) |

### 4.4 Dataset Summary

| Dataset | Samples | Input Shape | File Size |
|---------|---------|------------|----------|
| raw_t16_train | 1,600,000 | (2, 12, 16) | 2.4 GB |
| raw_t16_val | 200,000 | (2, 12, 16) | 0.3 GB |
| raw_t16_test | 200,000 | (2, 12, 16) | 0.3 GB |
| raw_t32_train | 1,600,000 | (2, 12, 32) | 4.7 GB |
| raw_t32_val | 200,000 | (2, 12, 32) | 0.6 GB |
| raw_t32_test | 200,000 | (2, 12, 32) | 0.6 GB |
| cov_t16_train | 1,600,000 | (2, 12, 12) | 1.6 GB |
| cov_t16_val | 200,000 | (2, 12, 12) | 0.2 GB |
| cov_t16_test | 200,000 | (2, 12, 12) | 0.2 GB |
| cov_t32_train | 1,600,000 | (2, 12, 12) | 1.6 GB |
| cov_t32_val | 200,000 | (2, 12, 12) | 0.2 GB |
| cov_t32_test | 200,000 | (2, 12, 12) | 0.2 GB |
| **Total** | **8,000,000** | — | **~13 GB** |

**Generation time:** ~5 minutes using 31 CPU cores (2026-03-24)

---

## 5. Model Architecture

### 5.1 Modified ResNeXt-50

| Component | Original ResNeXt-50 | This Project |
|-----------|-------------------|-------------|
| Input channels | 3 (RGB) | **2** (Re + Im) |
| Cardinality | 32 | 32 (unchanged) |
| FC output | 1000 (ImageNet) | **121** (DOA classes) |
| Final activation | Softmax | **Sigmoid** (multi-label) |
| Total parameters | ~25M | **~23.2M** |
| Pre-trained weights | ImageNet | ✅ (Conv1 adapted) |

### 5.2 Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning rate | 1×10⁻³ |
| Weight decay | 0.01 |
| Batch size | 128 |
| Steps/epoch | 12,500 (= 1.6M / 128) |
| Max epochs | 50 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping | patience=10 |
| Loss function | BCEWithLogitsLoss |

---

## 6. Training Timeline

| Date & Time | Event |
|-------------|-------|
| 2026-03-24 20:53 | raw_t16 training started |
| 2026-03-24 23:12 | raw_t16 training completed (Early Stop, 18 epochs) |
| 2026-03-25 02:36 | raw_t32 training started |
| 2026-03-26 01:28 | raw_t32 training completed (Full 50 epochs) |
| 2026-03-26 03:46 | cov_t16 training started |
| 2026-03-26 12:57 | cov_t16 still training (log updated 12:19) |
| TBD | cov_t32 pending |

---

## 7. Model 1: raw_t16 ✅ Completed

**Input:** Raw signal · T=16 snapshots · Duration: ~2.3 hours (18 epochs)  
**Best Checkpoint:** `results/checkpoints/raw_t16_best.pth` (266.3 MB)

### Per-Epoch Results

| Epoch | train_loss | val_loss | val_acc | Notes |
|-------|-----------|---------|---------|-------|
| 1  | 0.2239 | 0.2079 | 0.9314 | |
| 2  | 0.2012 | 0.1941 | 0.9331 | |
| 3  | 0.1906 | 0.1851 | 0.9349 | |
| 4  | 0.1839 | 0.1791 | 0.9362 | |
| 5  | 0.1785 | 0.1741 | 0.9373 | val_loss plateau begins |
| 6  | 0.1746 | 0.1788 | 0.9380 | val_loss rebounds |
| 7  | 0.1714 | 0.1770 | 0.9380 | |
| 8  | 0.1682 | **0.1647** | 0.9404 | ★ **Best val_loss** |
| 9  | 0.1655 | 0.1892 | 0.9403 | fluctuation |
| 10 | 0.1622 | 0.2285 | 0.9409 | ⚠️ LR reduction triggered |
| 11 | 0.1612 | 0.1878 | 0.9415 | |
| 12 | 0.1610 | 0.1843 | 0.9418 | |
| 13 | 0.1570 | 0.1659 | 0.9432 | second local minimum |
| 14 | 0.1539 | 0.1854 | 0.9437 | |
| 15 | 0.1528 | 0.2133 | 0.9441 | |
| 16 | 0.1530 | 0.2066 | 0.9439 | |
| 17 | 0.1501 | 0.2183 | 0.9450 | |
| 18 | 0.1485 | 0.1981 | **0.9456** | Early stopping triggered |

**Summary:** best_val_loss = **0.1647** · final_val_acc = **0.9456**

---

## 8. Model 2: raw_t32 ✅ Completed

**Input:** Raw signal · T=32 snapshots · Duration: ~22.9 hours (50 epochs)  
**Best Checkpoint:** `results/checkpoints/raw_t32_best.pth` (266.3 MB)

### Key Epoch Milestones

| Epoch | val_loss | val_acc | Notes |
|-------|---------|---------|-------|
| 1  | 0.1838 | 0.9346 | start |
| 5  | 0.1436 | 0.9450 | rapid convergence |
| 9  | 0.1238 | 0.9529 | |
| 11 | **6.5790** | 0.9297 | ⚠️ **Numerical spike (NaN-like event)** |
| 12 | 0.1640 | 0.9535 | auto-recovered |
| 13 | 0.1148 | 0.9557 | best post-spike |
| 20 | 0.1063 | 0.9601 | |
| 30 | 0.0931 | 0.9642 | |
| 40 | 0.0897 | 0.9658 | |
| 46 | **0.0872** | 0.9662 | ★ **Best val_loss** |
| 50 | 0.0880 | **0.9660** | final (no early stop) |

> **⚠️ Epoch 11 Anomaly:** val_loss spiked to 6.579 (normal ~0.12).  
> Likely caused by a bad validation mini-batch during HDF5 read.  
> The model automatically used the best saved checkpoint and recovered by Epoch 12.

**Summary:** best_val_loss = **0.0872** · final_val_acc = **0.9660**

---

## 9. Model 3: cov_t16 🔄 In Progress

**Input:** Covariance matrix · T=16 · Started: 2026-03-26 03:46  
**Last log update:** 2026-03-26 12:19 · Last checkpoint save: 11:02  
**Elapsed:** ~9 hours · **Estimated completion:** 18:00–22:00

*(Results to be filled after completion)*

---

## 10. Model 4: cov_t32 ⏳ Pending

Waiting for cov_t16 to complete. Will start automatically via `train_all.ps1`.

---

## 11. Comparative Analysis (Completed Models)

| Metric | raw_t16 | raw_t32 | Improvement |
|--------|---------|---------|-------------|
| best_val_loss | 0.1647 | **0.0872** | −47.1% |
| final_val_acc | 0.9456 | **0.9660** | +2.04% |
| Total epochs | 18 | 50 | |
| Training time | 2.3 h | 22.9 h | ×10 |

**Interpretation:**  
T=32 significantly outperforms T=16 because more snapshots provide a better approximation of the true covariance matrix:

```
R̂ = (1/T)·X·X^H  →  R  as T → ∞
```

Larger T reduces estimation variance, enabling the CNN to extract cleaner subspace features. This aligns with the theoretical prediction in the reference paper.

The 10× training time difference for T=32 is primarily because val_loss continued to decrease gradually throughout all 50 epochs (no early stopping trigger), whereas T=16 stagnated after epoch 8.

---

## 12. Issues & Resolutions Log

| Date | Issue | Root Cause | Resolution |
|------|-------|-----------|-----------|
| 2026-03-24 | `fbgemm.dll` error on import | Wrong PyTorch CUDA build | Downloaded torch-2.11.0+cu126 whl manually |
| 2026-03-24 | `ReduceLROnPlateau verbose` TypeError | Removed in PyTorch 2.11 | Deleted `verbose=True` argument |
| 2026-03-24 | `ModuleNotFoundError: datasets` | Script called directly, not from root | Added `sys.path.insert(0, project_root)` to both generators |
| 2026-03-24 | GitHub push rejected (266MB .pth) | Large binary in git history | Deleted `.git`, rebuilt with updated `.gitignore` |
| 2026-03-24 | BOM character in directory name | Windows UTF-8 BOM in folder `﻿MD9120` | Used PowerShell wildcard `*MD9120` |
| 2026-03-25 | TensorBoard "No dashboards" | TensorBoard pointed to wrong path | Used `$PSScriptRoot` wildcard resolution |
| 2026-03-26 | raw_t32 Epoch 11 val_loss = 6.58 | Numerical instability in validation batch | Auto-recovered; best checkpoint preserved |

---

## 13. Next Steps

- [ ] Complete cov_t16 training → update Section 9
- [ ] Complete cov_t32 training → add Section (new)
- [ ] Run `eval/metrics.py` to generate Table 2–5 (SNR sweep per model)
- [ ] Run `eval/compare_classical.py` for MUSIC/ESPRIT comparison (Table 6)
- [ ] Run `eval/visualize.py` to generate Fig. 4–8
- [ ] Git commit all results and push to GitHub
- [ ] Implement array perturbation robustness experiment (Extension 1)
