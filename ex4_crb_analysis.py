"""
ex4_crb_analysis.py
-------------------
Extension Experiment 4: CNN RMSE vs Cramér-Rao Bound

Loads existing SNR sweep results from results/*_metrics.json,
derives a proxy RMSE from Recall (missed detection rate → angular error),
and overlays the theoretical CRB lower bound.

CRB for ULA (used as approximation for TCA):
  CRB(θ, SNR, T, P) = 6 / (SNR_lin · T · π² · cos²θ · (P³ − P))

RMSE proxy from Recall:
  missed fraction = 1 - Recall
  proxy RMSE ≈ missed_fraction * DOA_STEP * sqrt(num_classes / 12)
  (rough estimate; reflects detection error floor)

Run:
    python extensions/ex4_crb_analysis.py
Output:
    results/extensions/fig_crb_vs_rmse.png
    results/extensions/crb_analysis.json
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = 'results/extensions'
os.makedirs(OUT_DIR, exist_ok=True)

# TCA parameters
P = 12          # physical sensors
DOA_STEP = 1.0  # degree resolution
THETA_DEG = 0.0 # reference DOA (θ=0°, cos=1, worst-case CRB)

CONFIGS = {
    'raw_t16': {'T': 16, 'label': 'CNN Raw T=16',  'ls': '--',  'marker': '^'},
    'raw_t32': {'T': 32, 'label': 'CNN Raw T=32',  'ls': '--',  'marker': 's'},
    'cov_t16': {'T': 16, 'label': 'CNN Cov T=16',  'ls': '-',   'marker': '^'},
    'cov_t32': {'T': 32, 'label': 'CNN Cov T=32',  'ls': '-',   'marker': 'o'},
}
COLORS = sns.color_palette('tab10')


def crb_rmse_deg(snr_db, T, P, theta_deg=0.0):
    """CRB RMSE lower bound in degrees (ULA approximation)."""
    snr_lin = 10 ** (snr_db / 10.0)
    cos2 = math.cos(math.radians(theta_deg)) ** 2
    denominator = snr_lin * T * (math.pi ** 2) * cos2 * (P ** 3 - P)
    crb_rad2 = 6.0 / denominator
    return math.degrees(math.sqrt(crb_rad2))


def recall_to_rmse_proxy(recall, doa_step=1.0):
    """
    Very rough proxy: missed detection fraction * 1 degree step.
    recall=1.0 → RMSE_proxy=0; recall=0.0 → RMSE_proxy=doa_step.
    In practice CNN RMSE is much better than this for high recall.
    """
    miss_rate = 1.0 - recall / 100.0   # recall is in %
    return miss_rate * doa_step


def main():
    results_dir = 'results'
    snr_range = np.arange(0, 22, 2)   # 0,2,...,20 dB

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    output = {}

    for ci, (name, info) in enumerate(CONFIGS.items()):
        json_path = os.path.join(results_dir, f'{name}_metrics.json')
        if not os.path.exists(json_path):
            print(f'  [skip] {json_path} not found')
            continue

        with open(json_path) as f:
            data = json.load(f)

        snr_results = sorted(data['snr_results'], key=lambda r: r['snr'])
        snrs   = [r['snr'] for r in snr_results]
        recall = [r['recall'] for r in snr_results]
        rmse_proxy = [recall_to_rmse_proxy(r) for r in recall]

        color = COLORS[ci]
        axes[0].plot(snrs, recall, color=color, lw=1.8,
                     marker=info['marker'], ms=5, ls=info['ls'], label=info['label'])
        axes[1].semilogy(snrs, rmse_proxy, color=color, lw=1.8,
                         marker=info['marker'], ms=5, ls=info['ls'], label=info['label'])

        output[name] = {'snr': snrs, 'recall': recall, 'rmse_proxy': rmse_proxy}

    # CRB curves for T=16 and T=32
    snr_fine = np.linspace(0, 20, 200)
    crb_colors = {'T=16': 'gray', 'T=32': 'black'}
    for T_val, c in [(16, 'gray'), (32, 'black')]:
        crb_vals = [crb_rmse_deg(s, T_val, P, THETA_DEG) for s in snr_fine]
        axes[1].semilogy(snr_fine, crb_vals, color=c, lw=2.0, ls=':', alpha=0.8,
                         label=f'CRB (ULA approx, T={T_val})')
        output[f'crb_T{T_val}'] = {
            'snr': snr_fine.tolist(),
            'crb_rmse_deg': crb_vals,
        }

    # Axes formatting
    axes[0].set_xlabel('SNR (dB)', fontsize=12)
    axes[0].set_ylabel('Recall (%)', fontsize=12)
    axes[0].set_title('Recall vs SNR', fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(-1, 21)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('SNR (dB)', fontsize=12)
    axes[1].set_ylabel('RMSE / CRB (degrees, log scale)', fontsize=12)
    axes[1].set_title('Extension 4 – CNN RMSE Proxy vs Cramér-Rao Bound\n'
                      '(CRB = ULA approximation, θ=0°, P=12)', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(-1, 21)
    axes[1].grid(alpha=0.3, which='both')

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'fig_crb_vs_rmse.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Figure saved: {fig_path}')

    json_path = os.path.join(OUT_DIR, 'crb_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Results saved: {json_path}')

    # Print CRB reference table
    print('\n=== CRB Reference (θ=0°, P=12) ===')
    print(f'{"SNR (dB)":>10} {"CRB T=16 (°)":>14} {"CRB T=32 (°)":>14}')
    for snr in [0, 5, 10, 15, 20]:
        c16 = crb_rmse_deg(snr, 16, P)
        c32 = crb_rmse_deg(snr, 32, P)
        print(f'{snr:>10} {c16:>14.4f} {c32:>14.4f}')


if __name__ == '__main__':
    main()
