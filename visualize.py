"""
visualize.py
------------
Automated plot generation for all paper figures and supplementary figures.

Generates (all at 300 DPI PNG + optional PDF):
  Fig.4 : Training / Validation loss curves for 4 models
  Fig.5 : Performance metrics vs snapshot count (T=16 vs T=32)
  Fig.6 : CNN vs MUSIC vs ESPRIT -- Accuracy vs SNR (extension experiment)
  Fig.7 : TCA array layout
  Fig.8 : Spatial spectrum of MUSIC for one sample

Usage:
    python eval/visualize.py --all --results results/
    python eval/visualize.py --figure 4 --results results/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # headless rendering (no display needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from datasets.array_geometry import get_tca_positions

# ── Style settings ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 12,
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 12,
    'legend.fontsize'  : 11,
    'figure.dpi'       : 300,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
})
COLORS = sns.color_palette('tab10')
FIGDIR = 'results/figures'


def ensure_figdir(d: str = FIGDIR) -> str:
    os.makedirs(d, exist_ok=True)
    return d


# ── Fig.4: Loss curves ─────────────────────────────────────────────────────────

def plot_loss_curves(results_dir: str = 'results', figdir: str = FIGDIR) -> None:
    """
    Plot train/val loss curves for all 4 model configurations (Fig.4).
    """
    ensure_figdir(figdir)
    configs = {
        'raw_t16': 'Raw, T=16',
        'raw_t32': 'Raw, T=32',
        'cov_t16': 'Cov, T=16',
        'cov_t32': 'Cov, T=32',
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    axes = axes.ravel()

    for ax, (cfg_name, label) in zip(axes, configs.items()):
        json_path = os.path.join(results_dir, f'{cfg_name}_metrics.json')
        if not os.path.isfile(json_path):
            ax.set_title(f'{label} (no data)')
            continue

        with open(json_path) as f:
            data = json.load(f)
        hist = data.get('history', {})
        if not hist:
            # Try loading from checkpoint directly
            ckpt_path = os.path.join(results_dir, 'checkpoints',
                                     f'{cfg_name}_best.pth')
            if os.path.isfile(ckpt_path):
                import torch
                ckpt = torch.load(ckpt_path, map_location='cpu')
                hist = ckpt.get('history', {})

        if not hist or 'train_loss' not in hist:
            ax.set_title(f'{label} (no history)')
            continue

        epochs = range(1, len(hist['train_loss']) + 1)
        ax.plot(epochs, hist['train_loss'], color=COLORS[0], label='Train Loss',  lw=1.5)
        ax.plot(epochs, hist['val_loss'],   color=COLORS[1], label='Val Loss',
                lw=1.5, linestyle='--')
        ax.set_title(label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('BCE Loss')
        ax.legend()

    fig.suptitle('Fig. 4 – Training and Validation Loss Curves', fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(figdir, 'fig4_loss_curves.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


# ── Fig.5: Performance vs T ────────────────────────────────────────────────────

def plot_performance_vs_T(results_dir: str = 'results', figdir: str = FIGDIR,
                           eval_snr: float = 10.0) -> None:
    """
    Bar chart / grouped comparison: T=16 vs T=32 for raw and cov inputs (Fig.5).
    """
    ensure_figdir(figdir)
    configs = ['raw_t16', 'raw_t32', 'cov_t16', 'cov_t32']
    metrics = ['accuracy', 'precision', 'recall', 'specificity']

    # Extract metrics at target SNR
    rows = []
    for cfg_name in configs:
        json_path = os.path.join(results_dir, f'{cfg_name}_metrics.json')
        if not os.path.isfile(json_path):
            continue
        with open(json_path) as f:
            data = json.load(f)
        snr_results = data.get('snr_results', [])
        for r in snr_results:
            if abs(r['snr'] - eval_snr) < 0.1:
                itype = 'Raw' if 'raw' in cfg_name else 'Cov'
                T_val = int(cfg_name.split('t')[-1])
                rows.append({'Input': itype, 'T': T_val,
                             **{m: r[m] for m in metrics}})

    if not rows:
        print('  [skip Fig.5] No results found at SNR={eval_snr}dB')
        return

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)

    for ax, metric in zip(axes, metrics):
        sub = df.pivot(index='T', columns='Input', values=metric)
        sub.plot(kind='bar', ax=ax, color=[COLORS[0], COLORS[2]], edgecolor='k',
                 width=0.6, legend=(metric == 'accuracy'))
        ax.set_title(metric.capitalize())
        ax.set_xlabel('Snapshots T')
        ax.set_ylabel('Value (%)')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['T=16', 'T=32'], rotation=0)

    fig.suptitle(f'Fig. 5 – Performance vs Snapshot Count  (SNR={eval_snr}dB)',
                 fontsize=14)
    plt.tight_layout()
    out = os.path.join(figdir, 'fig5_performance_vs_T.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


# ── Fig.6: CNN vs Classical ────────────────────────────────────────────────────

def plot_classical_comparison(results_dir: str = 'results',
                               figdir: str = FIGDIR) -> None:
    """
    Plot Accuracy vs SNR for CNN (4 configs) + MUSIC + ESPRIT (Fig.6).
    """
    ensure_figdir(figdir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Load classical results
    classical_path = os.path.join(results_dir, 'classical_metrics.json')
    if os.path.isfile(classical_path):
        with open(classical_path) as f:
            classical_data = json.load(f)
        df_cls = pd.DataFrame(classical_data)

        for ci, method in enumerate(['MUSIC', 'ESPRIT']):
            sub = df_cls[df_cls['method'] == method].sort_values('snr')
            axes[0].plot(sub['snr'], sub['accuracy'],
                         marker='s', label=method, color=COLORS[ci + 4],
                         lw=1.5, linestyle='--')
            axes[1].plot(sub['snr'], sub['recall'],
                         marker='s', label=method, color=COLORS[ci + 4],
                         lw=1.5, linestyle='--')

    # Load CNN results
    cnn_configs = {
        'raw_t16': ('Raw T=16', COLORS[0]),
        'raw_t32': ('Raw T=32', COLORS[1]),
        'cov_t16': ('Cov T=16', COLORS[2]),
        'cov_t32': ('Cov T=32', COLORS[3]),
    }
    for cfg_name, (label, color) in cnn_configs.items():
        json_path = os.path.join(results_dir, f'{cfg_name}_metrics.json')
        if not os.path.isfile(json_path):
            continue
        with open(json_path) as f:
            data = json.load(f)
        rows = sorted(data.get('snr_results', []), key=lambda r: r['snr'])
        snrs = [r['snr'] for r in rows]
        axes[0].plot(snrs, [r['accuracy'] for r in rows],
                     marker='o', label=f'CNN {label}', color=color, lw=1.5)
        axes[1].plot(snrs, [r['recall'] for r in rows],
                     marker='o', label=f'CNN {label}', color=color, lw=1.5)

    for ax, metric_name in zip(axes, ['Accuracy (%)', 'Recall (%)']):
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs SNR')
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    fig.suptitle('Fig. 6 – CNN vs MUSIC vs ESPRIT: Performance vs SNR', fontsize=14)
    plt.tight_layout()
    out = os.path.join(figdir, 'fig6_cnn_vs_classical.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


# ── Fig.7: TCA array layout ────────────────────────────────────────────────────

def plot_tca_layout(M: int = 5, N: int = 6, figdir: str = FIGDIR) -> None:
    """Visualise TCA sensor positions on a line."""
    ensure_figdir(figdir)
    positions = get_tca_positions(M, N)

    sub1 = set(k * N for k in range(M))
    sub2 = set(k * M for k in range(N + 2))

    fig, ax = plt.subplots(figsize=(11, 2))
    for p in positions:
        in1 = p in sub1
        in2 = p in sub2
        if in1 and in2:
            color, zorder = 'purple', 4
        elif in1:
            color, zorder = 'royalblue', 3
        else:
            color, zorder = 'tomato', 3
        ax.plot(p, 0, 'o', color=color, markersize=13, markeredgecolor='k',
                zorder=zorder)
        ax.text(p, -0.12, str(int(p)), ha='center', va='top', fontsize=9)

    ax.set_xlim(-2, max(positions) + 2)
    ax.set_ylim(-0.5, 0.4)
    ax.set_xlabel('Position  (unit = d = λ/2)')
    ax.set_title(
        f'Thinned Coprime Array  (M={M}, N={N})  →  {len(positions)} sensors\n'
        'Blue=Sub-array 1   Red=Sub-array 2   Purple=shared'
    )
    ax.yaxis.set_visible(False)
    plt.tight_layout()
    out = os.path.join(figdir, 'fig7_tca_layout.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


# ── Fig.8: MUSIC spatial spectrum ─────────────────────────────────────────────

def plot_music_spectrum(snr_db: float = 10.0, K: int = 3,
                         figdir: str = FIGDIR) -> None:
    """
    Generate and plot a MUSIC spatial spectrum for a sample scenario.
    """
    ensure_figdir(figdir)
    from datasets.array_geometry import get_tca_positions, get_steering_matrix
    from eval.compare_classical import simulate_signal, sample_covariance, music

    positions = get_tca_positions()
    rng = np.random.default_rng(42)
    T   = 32
    X, thetas, _ = simulate_signal(positions, T, snr_db, K=K, rng=rng)
    R   = sample_covariance(X)

    search_grid = np.arange(-90, 91, 0.1)
    _, spectrum  = music(R, K, positions, search_grid=search_grid,
                         return_spectrum=True)

    # Normalise to dB
    spectrum_db = 10 * np.log10(spectrum / spectrum.max() + 1e-15)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(search_grid, spectrum_db, color=COLORS[0], lw=1.5, label='MUSIC spectrum')
    for th in thetas:
        ax.axvline(th, color='tomato', lw=1.5, linestyle='--',
                   label=f'True DOA={th:.0f}°')
    ax.set_xlabel('DOA (degrees)')
    ax.set_ylabel('Normalised Power (dB)')
    ax.set_title(f'Fig. 8 – MUSIC Spatial Spectrum  (K={K}, SNR={snr_db}dB, T={T})')
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(figdir, 'fig8_music_spectrum.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


# ── Generate all figures ───────────────────────────────────────────────────────

def generate_all(results_dir: str = 'results') -> None:
    figdir = os.path.join(results_dir, 'figures')
    print('Generating all figures...')
    plot_tca_layout(figdir=figdir)
    plot_loss_curves(results_dir, figdir)
    plot_performance_vs_T(results_dir, figdir)
    plot_classical_comparison(results_dir, figdir)
    plot_music_spectrum(figdir=figdir)
    print(f'\nAll figures saved to: {figdir}')


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--all',     action='store_true',
                        help='Generate all figures')
    parser.add_argument('--figure',  type=int, default=None,
                        help='Generate specific figure: 4, 5, 6, 7, or 8')
    parser.add_argument('--results', type=str, default='results')
    args = parser.parse_args()

    figdir = os.path.join(args.results, 'figures')

    if args.all:
        generate_all(args.results)
    elif args.figure == 4:
        plot_loss_curves(args.results, figdir)
    elif args.figure == 5:
        plot_performance_vs_T(args.results, figdir)
    elif args.figure == 6:
        plot_classical_comparison(args.results, figdir)
    elif args.figure == 7:
        plot_tca_layout(figdir=figdir)
    elif args.figure == 8:
        plot_music_spectrum(figdir=figdir)
    else:
        print('Use --all or --figure [4|5|6|7|8]')
