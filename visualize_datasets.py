"""
visualize_datasets.py  (v2 — one figure per dataset)
------------------------------------------------------
Generates 4 separate figures, one per dataset configuration.

Output:
    results/figures/dataset_raw_t16.png
    results/figures/dataset_raw_t32.png
    results/figures/dataset_cov_t16.png
    results/figures/dataset_cov_t32.png

Usage:
    python datasets/visualize_datasets.py            # auto-detect HDF5
    python datasets/visualize_datasets.py --synthetic
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from datasets.array_geometry import get_tca_positions, get_steering_matrix

# ── Constants ─────────────────────────────────────────────────────────────────
DOA_MIN  = -60
DOA_MAX  =  60
DOA_STEP =   1
DOA_GRID = np.arange(DOA_MIN, DOA_MAX + DOA_STEP, DOA_STEP)
OUT_DIR  = 'results/figures'
os.makedirs(OUT_DIR, exist_ok=True)

TCA_POSITIONS = get_tca_positions(M=5, N=6)
P = len(TCA_POSITIONS)
BG   = '#0d1117'
CARD = '#161b22'
GRID_COLOR = '#30363d'


# ── Signal simulation ─────────────────────────────────────────────────────────

def make_synthetic_sample(T: int, snr_db: float = 10.0, K: int = 3, seed: int = 42):
    rng    = np.random.default_rng(seed)
    idx    = rng.choice(len(DOA_GRID), size=K, replace=False)
    thetas = DOA_GRID[idx]
    A      = get_steering_matrix(thetas, TCA_POSITIONS)
    s      = (rng.standard_normal((K, T)) + 1j * rng.standard_normal((K, T))) / np.sqrt(2)
    X_c    = A @ s
    sp     = np.mean(np.abs(X_c) ** 2)
    ns     = np.sqrt(sp / (10 ** (snr_db / 10)) / 2)
    noise  = ns * (rng.standard_normal((P, T)) + 1j * rng.standard_normal((P, T)))
    X      = X_c + noise
    R      = (X @ X.conj().T) / T
    x_raw  = np.stack([X.real, X.imag], axis=0).astype(np.float32)
    x_cov  = np.stack([R.real, R.imag], axis=0).astype(np.float32)
    label  = np.zeros(len(DOA_GRID), dtype=np.float32)
    label[idx] = 1.0
    return x_raw, x_cov, label, thetas


def try_load_h5(path, n=3):
    try:
        import h5py
        if not os.path.exists(path): return None
        with h5py.File(path, 'r') as f:
            return f['X'][:n], f['Y'][:n]
    except Exception:
        return None


# ── Per-dataset figure ────────────────────────────────────────────────────────

def plot_one_dataset(key, title, h5_path, T, accent, synthetic, n_examples=3):
    """Render one rich figure for a single dataset configuration."""
    is_raw = 'raw' in key
    samples = []

    if not synthetic:
        result = try_load_h5(h5_path, n=n_examples)
        if result is not None:
            Xs, Ys = result
            for i in range(len(Xs)):
                xt = Xs[i]   # (2,P,T) or (2,P,P)
                lbl = Ys[i]
                thetas = DOA_GRID[lbl > 0.5]
                if is_raw:
                    x_raw = xt
                    Xc = xt[0] + 1j * xt[1]
                    R  = (Xc @ Xc.conj().T) / T
                    x_cov = np.stack([R.real, R.imag], axis=0)
                else:
                    x_raw = None
                    x_cov = xt
                samples.append((x_raw, x_cov, lbl, thetas))

    if not samples:
        for i in range(n_examples):
            xr, xc, lbl, th = make_synthetic_sample(T, K=np.random.randint(1,7), seed=42+i)
            if not is_raw: xr = None
            samples.append((xr, xc, lbl, th))

    n = len(samples)
    # Layout: n_examples columns
    # Row 0: raw signal (real) — only for raw; or placeholder for cov
    # Row 1: raw signal (imag) — only for raw
    # Row 2: cov real part
    # Row 3: cov imag part
    # Row 4: DOA label bar
    # Row 5 (shared): TCA geometry

    n_data_rows = 5
    fig = plt.figure(figsize=(7 * n, 22))
    fig.patch.set_facecolor(BG)

    outer = gridspec.GridSpec(n_data_rows + 1, n, figure=fig,
                               hspace=0.55, wspace=0.3,
                               height_ratios=[1, 1, 1, 1, 1.2, 0.7])

    row_titles = [
        'Re{X(t)} — Raw Signal (Real part)',
        'Im{X(t)} — Raw Signal (Imaginary part)',
        'Re{R̂} — Sample Covariance (Real part)',
        'Im{R̂} — Sample Covariance (Imaginary part)',
        'DOA Label  (121-dim multi-hot)',
    ]

    for col, (x_raw, x_cov, lbl, thetas) in enumerate(samples):
        K_used = int(lbl.sum())

        def make_ax(row):
            ax = fig.add_subplot(outer[row, col])
            ax.set_facecolor(CARD)
            ax.tick_params(colors='#aaa', labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)
            return ax

        # Row 0 — raw real
        ax0 = make_ax(0)
        if is_raw and x_raw is not None:
            im0 = ax0.imshow(x_raw[0], aspect='auto', cmap='RdBu_r',
                             interpolation='nearest')
            ax0.set_xlabel('Snapshot t', color='#aaa', fontsize=8)
            ax0.set_ylabel('Sensor index', color='#aaa', fontsize=8)
            cb = plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelcolor='#aaa', labelsize=7)
        else:
            ax0.text(0.5, 0.5, 'Not stored\n(Cov input discards raw X)',
                     ha='center', va='center', color='#666', fontsize=10,
                     transform=ax0.transAxes)
        if col == 0: ax0.set_ylabel(row_titles[0], color='#ccc', fontsize=8.5, labelpad=8)

        # Row 1 — raw imag
        ax1 = make_ax(1)
        if is_raw and x_raw is not None:
            im1 = ax1.imshow(x_raw[1], aspect='auto', cmap='RdBu_r',
                             interpolation='nearest')
            ax1.set_xlabel('Snapshot t', color='#aaa', fontsize=8)
            cb = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelcolor='#aaa', labelsize=7)
        else:
            ax1.text(0.5, 0.5, 'Not stored', ha='center', va='center',
                     color='#666', fontsize=10, transform=ax1.transAxes)
        if col == 0: ax1.set_ylabel(row_titles[1], color='#ccc', fontsize=8.5, labelpad=8)

        # Row 2 — cov real
        ax2 = make_ax(2)
        im2 = ax2.imshow(x_cov[0], aspect='auto', cmap='viridis',
                         interpolation='nearest')
        ax2.set_xlabel('Sensor j', color='#aaa', fontsize=8)
        ax2.set_ylabel('Sensor i', color='#aaa', fontsize=8)
        cb = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelcolor='#aaa', labelsize=7)
        if col == 0: ax2.set_ylabel(row_titles[2], color='#ccc', fontsize=8.5, labelpad=8)

        # Row 3 — cov imag
        ax3 = make_ax(3)
        im3 = ax3.imshow(x_cov[1], aspect='auto', cmap='PiYG',
                         interpolation='nearest')
        ax3.set_xlabel('Sensor j', color='#aaa', fontsize=8)
        cb = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelcolor='#aaa', labelsize=7)
        if col == 0: ax3.set_ylabel(row_titles[3], color='#ccc', fontsize=8.5, labelpad=8)

        # Row 4 — DOA label
        ax4 = make_ax(4)
        bar_c = [accent if lbl[i] > 0.5 else '#2a2d35' for i in range(121)]
        ax4.bar(DOA_GRID, lbl, color=bar_c, width=0.85)
        ax4.set_xlim(DOA_MIN - 1, DOA_MAX + 1)
        ax4.set_ylim(0, 1.5)
        ax4.set_xlabel('DOA angle (°)', color='#aaa', fontsize=8)
        for th in thetas:
            ax4.axvline(th, color=accent, alpha=0.7, lw=1.2, ls='--')
            ax4.text(th, 1.15, f'{th:.0f}°', ha='center', va='bottom',
                     color=accent, fontsize=8, fontweight='bold')
        ax4.set_title(f'Sample {col+1}  |  K={K_used} sources', color='#ccc',
                      fontsize=9, pad=4)
        ax4.tick_params(colors='#aaa', labelsize=8)
        ax4.set_xticks(range(-60, 61, 10))
        if col == 0: ax4.set_ylabel(row_titles[4], color='#ccc', fontsize=8.5, labelpad=8)

    # Row 5 — TCA geometry (shared, full width)
    ax5 = fig.add_subplot(outer[5, :])
    ax5.set_facecolor(CARD)
    for pos in TCA_POSITIONS:
        ax5.scatter(pos, 0, s=160, color=accent, zorder=3,
                    edgecolors='white', linewidths=0.8)
        ax5.text(pos, 0.22, f'{int(pos)}d', ha='center', fontsize=8, color='#ccc')
    ax5.axhline(0, color='#444', lw=1, zorder=1)

    # Sub-array annotations
    sa1 = [0, 5, 10, 15, 20, 25]
    sa2 = [6, 12]
    sa3 = [36, 42, 48, 54]
    for pos in sa1:
        ax5.scatter(pos, 0, s=200, color='#58a6ff', zorder=4, marker='^')
    for pos in sa2:
        ax5.scatter(pos, 0, s=200, color='#f78166', zorder=4, marker='s')
    for pos in sa3:
        ax5.scatter(pos, 0, s=200, color='#3fb950', zorder=4, marker='D')

    ax5.set_xlim(-3, 58)
    ax5.set_ylim(-0.6, 0.7)
    ax5.set_xlabel('Position (× λ/2)', color='#aaa', fontsize=9)
    ax5.set_yticks([])
    ax5.set_title(
        'TCA Array Geometry  |  M=5, N=6  |  P=12  |  Aperture=54d  '
        '|  ▲ SubArr1(step 5d)  ■ SubArr2(step 6d, short)  ◆ SubArr3(offset)',
        color='#ccc', fontsize=9, pad=6)
    for sp in ax5.spines.values(): sp.set_edgecolor(GRID_COLOR)

    fig.suptitle(
        f'{title}\n'
        f'Input shape: {"(2, 12, " + str(T) + ")" if is_raw else "(2, 12, 12)"}  |  '
        f'Label: (121,) multi-hot  |  DOA ∈ [−60°, +60°], step 1°  |  SNR=10 dB',
        color='white', fontsize=13, fontweight='bold', y=1.001
    )

    out = os.path.join(OUT_DIR, f'dataset_{key}.png')
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  Saved -> {out}')
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()

    configs = [
        ('raw_t16', 'Dataset 1: Raw Signal  T=16  (raw_t16)',
         'data/raw_t16_train.h5', 16, '#4c9be8'),
        ('raw_t32', 'Dataset 2: Raw Signal  T=32  (raw_t32)',
         'data/raw_t32_train.h5', 32, '#f0883e'),
        ('cov_t16', 'Dataset 3: Covariance Matrix  T=16  (cov_t16)',
         'data/cov_t16_train.h5', 16, '#56d364'),
        ('cov_t32', 'Dataset 4: Covariance Matrix  T=32  (cov_t32)',
         'data/cov_t32_train.h5', 32, '#bc8cff'),
    ]

    print(f'Generating {len(configs)} dataset figures...')
    for cfg in configs:
        plot_one_dataset(*cfg, synthetic=args.synthetic)
    print('Done.')


if __name__ == '__main__':
    main()
