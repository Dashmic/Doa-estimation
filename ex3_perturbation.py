"""
ex3_perturbation.py
-------------------
Extension Experiment 3: Array Position Perturbation Robustness

Tests how CNN (trained on nominal TCA) degrades when the physical
array positions are perturbed by Gaussian noise:

    p_perturbed = p_nominal + δ,   δ ~ N(0, ε²)

Perturbation levels: ε ∈ {0, 0.02, 0.05, 0.10, 0.20}  (in units of d = λ/2)

Compares:
  - CNN cov_t32  (full covariance input; most robust expected)
  - CNN raw_t32  (raw signal; may be more sensitive)

Run:
    python extensions/ex3_perturbation.py
Output:
    results/extensions/perturbation_results.json
    results/extensions/fig_perturbation.png
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from datasets.array_geometry import get_tca_positions
from datasets.generate_raw import simulate_snapshot
from datasets.generate_cov import simulate_cov
from models.resnext_doa import ResNeXtDOA

OUT_DIR  = 'results/extensions'
os.makedirs(OUT_DIR, exist_ok=True)

# Perturbation levels (σ as fraction of d = λ/2)
EPSILON_LIST = [0.0, 0.02, 0.05, 0.10, 0.20]
N_PER_EPS    = 5000       # samples per epsilon level
SNR_DB       = 10.0

MODELS = [
    ('cov_t32', 'configs/cov_t32.yaml', 'results/checkpoints/cov_t32_best.pth'),
    ('raw_t32', 'configs/raw_t32.yaml', 'results/checkpoints/raw_t32_best.pth'),
]


def compute_metrics(y_pred, y_true, threshold=0.5):
    pb = (y_pred >= threshold).astype(np.float32)
    gt = y_true.astype(np.float32)
    eps = 1e-8
    TP = (pb * gt).sum(axis=0)
    FP = (pb * (1 - gt)).sum(axis=0)
    TN = ((1 - pb) * (1 - gt)).sum(axis=0)
    FN = ((1 - pb) * gt).sum(axis=0)
    acc  = float(((TP + TN) / (TP + FP + TN + FN + eps)).mean())
    prec = float((TP / (TP + FP + eps)).mean())
    rec  = float((TP / (TP + FN + eps)).mean())
    return {'accuracy': acc * 100, 'precision': prec * 100, 'recall': rec * 100}


def generate_perturbed_batch(itype, T, epsilon, n_samples, rng):
    """Generate n_samples with perturbed array positions."""
    positions_nominal = get_tca_positions()
    X_list, Y_list = [], []
    for _ in range(n_samples):
        # Perturb positions for this sample
        if epsilon > 0:
            delta = rng.normal(0, epsilon, len(positions_nominal))
            positions_perturbed = positions_nominal + delta
        else:
            positions_perturbed = positions_nominal

        if itype == 'raw':
            # Need custom simulate with perturbed positions
            x, y = _simulate_raw_perturbed(positions_perturbed, T, SNR_DB, rng)
        else:
            x, y = _simulate_cov_perturbed(positions_perturbed, T, SNR_DB, rng)
        X_list.append(x)
        Y_list.append(y)
    return np.stack(X_list), np.stack(Y_list)


# ── DOA Grid constants (same across all configs) ───────────────────────────
_DOA_MIN     = -60
_DOA_MAX     =  60
_DOA_STEP    =   1
_NUM_CLASSES = 121
_K_MIN       =   1
_K_MAX       =  16
_DOA_GRID    = np.arange(_DOA_MIN, _DOA_MAX + _DOA_STEP, _DOA_STEP)


def _simulate_raw_perturbed(positions, T, snr_db, rng):
    """Simulate raw signal with custom (perturbed) array positions."""
    K = rng.integers(_K_MIN, _K_MAX + 1)
    theta_idx = rng.choice(_NUM_CLASSES, K, replace=False)
    thetas = _DOA_GRID[theta_idx]
    label = np.zeros(_NUM_CLASSES, dtype=np.float32)
    label[theta_idx] = 1.0

    A = np.exp(1j * np.pi * np.outer(positions, np.sin(np.deg2rad(thetas))))
    S = (rng.standard_normal((K, T)) + 1j * rng.standard_normal((K, T))) / np.sqrt(2)
    Xclean = A @ S
    sig_power = np.mean(np.abs(Xclean) ** 2)
    snr_lin   = 10 ** (snr_db / 10)
    noise_std = np.sqrt(sig_power / snr_lin / 2)
    noise = noise_std * (rng.standard_normal(Xclean.shape) + 1j * rng.standard_normal(Xclean.shape))
    X = Xclean + noise
    out = np.stack([X.real, X.imag], axis=0).astype(np.float32)  # (2, P, T)
    return out, label


def _simulate_cov_perturbed(positions, T, snr_db, rng):
    """Simulate covariance matrix with custom (perturbed) array positions."""
    K = rng.integers(_K_MIN, _K_MAX + 1)
    theta_idx = rng.choice(_NUM_CLASSES, K, replace=False)
    thetas = _DOA_GRID[theta_idx]
    label = np.zeros(_NUM_CLASSES, dtype=np.float32)
    label[theta_idx] = 1.0

    A = np.exp(1j * np.pi * np.outer(positions, np.sin(np.deg2rad(thetas))))
    S = (rng.standard_normal((K, T)) + 1j * rng.standard_normal((K, T))) / np.sqrt(2)
    Xclean = A @ S
    sig_power = np.mean(np.abs(Xclean) ** 2)
    snr_lin   = 10 ** (snr_db / 10)
    noise_std = np.sqrt(sig_power / snr_lin / 2)
    noise = noise_std * (rng.standard_normal(Xclean.shape) + 1j * rng.standard_normal(Xclean.shape))
    X = Xclean + noise
    R = (X @ X.conj().T) / T
    out = np.stack([R.real, R.imag], axis=0).astype(np.float32)  # (2, P, P)
    return out, label


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    rng = np.random.default_rng(42)

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = {'cov_t32': '#d62728', 'raw_t32': '#1f77b4'}
    markers = {'cov_t32': 'o', 'raw_t32': 's'}

    for name, cfg_path, ckpt_path in MODELS:
        print(f'\n[{name}]')
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        model = ResNeXtDOA(
            num_classes=cfg['num_classes'],
            input_type=cfg['input_type'],
            pretrained=False,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        rows = []
        for eps in EPSILON_LIST:
            print(f'  ε={eps:.2f}  generating {N_PER_EPS} samples...')
            X_arr, Y_arr = generate_perturbed_batch(
                cfg['input_type'], cfg['T'], eps, N_PER_EPS, rng
            )
            X_tensor = torch.from_numpy(X_arr)
            preds = []
            with torch.no_grad():
                for i in range(0, len(X_tensor), 512):
                    batch = X_tensor[i:i+512].to(device)
                    preds.append(model(batch).cpu().numpy())
            y_pred = np.concatenate(preds)
            m = compute_metrics(y_pred, Y_arr)
            m['epsilon'] = eps
            rows.append(m)
            print(f'    acc={m["accuracy"]:.2f}%  prec={m["precision"]:.2f}%  rec={m["recall"]:.2f}%')

        results[name] = rows

        eps_vals = [r['epsilon'] for r in rows]
        for ax, metric in zip(axes, ['accuracy', 'precision', 'recall']):
            vals = [r[metric] for r in rows]
            ax.plot(eps_vals, vals, color=colors[name], marker=markers[name],
                    lw=2.0, ms=7, label=name)

    for ax, metric in zip(axes, ['Accuracy', 'Precision', 'Recall']):
        ax.set_xlabel('Perturbation ε (× d = λ/2)', fontsize=11)
        ax.set_ylabel(f'{metric} (%)', fontsize=11)
        ax.set_title(metric, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle('Extension 3 – Array Position Perturbation Robustness\n'
                 f'(N={N_PER_EPS} samples/ε, SNR={SNR_DB}dB)', fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'fig_perturbation.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\nFigure saved: {fig_path}')

    json_path = os.path.join(OUT_DIR, 'perturbation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved: {json_path}')

    # Print LaTeX table
    print('\n=== Perturbation Table (Accuracy %) ===')
    print(f'{"ε":>6}', end='')
    for name, _, _ in MODELS:
        print(f'  {name:>10}', end='')
    print()
    for i, eps in enumerate(EPSILON_LIST):
        print(f'{eps:>6.2f}', end='')
        for name, _, _ in MODELS:
            print(f'  {results[name][i]["accuracy"]:>10.2f}', end='')
        print()


if __name__ == '__main__':
    main()
