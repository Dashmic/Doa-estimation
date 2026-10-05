"""
ex2_threshold_opt.py
--------------------
Extension Experiment 2: Optimal Decision Threshold Search

For multi-label DOA classification, the default threshold=0.5 is suboptimal
because labels are sparse (K sources active out of 121 classes).

Search threshold in [0.30, 0.95] on validation set for each of the 4 models.
Reports F1, Accuracy, Precision, Recall at each threshold and plots curves.

Run:
    python extensions/ex2_threshold_opt.py
Output:
    results/extensions/threshold_opt.json
    results/extensions/fig_threshold_opt.png
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
from datasets.data_loader import get_dataloaders
from models.resnext_doa import ResNeXtDOA

OUT_DIR = 'results/extensions'
os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = [
    ('raw_t16', 'configs/raw_t16.yaml', 'results/checkpoints/raw_t16_best.pth'),
    ('raw_t32', 'configs/raw_t32.yaml', 'results/checkpoints/raw_t32_best.pth'),
    ('cov_t16', 'configs/cov_t16.yaml', 'results/checkpoints/cov_t16_best.pth'),
    ('cov_t32', 'configs/cov_t32.yaml', 'results/checkpoints/cov_t32_best.pth'),
]
THRESHOLDS = np.arange(0.30, 0.96, 0.05)


def collect_val_predictions(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, Y in tqdm(loader, desc='  Collecting val preds', leave=False):
            out = model(X.to(device)).cpu().numpy()
            preds.append(out)
            labels.append(Y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def metrics_at_threshold(y_pred, y_true, thr):
    pb = (y_pred >= thr).astype(np.float32)
    gt = y_true.astype(np.float32)
    eps = 1e-8
    TP = (pb * gt).sum(axis=0)
    FP = (pb * (1 - gt)).sum(axis=0)
    TN = ((1 - pb) * (1 - gt)).sum(axis=0)
    FN = ((1 - pb) * gt).sum(axis=0)
    acc  = float(((TP + TN) / (TP + FP + TN + FN + eps)).mean())
    prec = float((TP / (TP + FP + eps)).mean())
    rec  = float((TP / (TP + FN + eps)).mean())
    f1   = 2 * prec * rec / (prec + rec + eps)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    colors = {'accuracy': '#1f77b4', 'precision': '#ff7f0e',
              'recall': '#2ca02c', 'f1': '#d62728'}

    for ax, (name, cfg_path, ckpt_path) in zip(axes, CONFIGS):
        print(f'\n[{name}] Loading model...')
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        model = ResNeXtDOA(
            num_classes=cfg['num_classes'],
            input_type=cfg['input_type'],
            pretrained=False,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])

        loaders = get_dataloaders(
            train_h5=cfg['train_h5'], val_h5=cfg['val_h5'], test_h5=cfg['test_h5'],
            input_type=cfg['input_type'], batch_size=512,
            num_workers=cfg['num_workers'], pin_memory=(device.type == 'cuda'),
        )
        y_pred, y_true = collect_val_predictions(model, loaders['val'], device)

        rows = []
        for thr in THRESHOLDS:
            m = metrics_at_threshold(y_pred, y_true, thr)
            m['threshold'] = float(round(thr, 2))
            rows.append(m)
            print(f'  thr={thr:.2f}  acc={m["accuracy"]*100:.2f}%  '
                  f'prec={m["precision"]*100:.2f}%  rec={m["recall"]*100:.2f}%  '
                  f'f1={m["f1"]*100:.2f}%')

        results[name] = rows
        best_f1 = max(rows, key=lambda r: r['f1'])
        print(f'  >> BEST threshold={best_f1["threshold"]:.2f}  '
              f'F1={best_f1["f1"]*100:.2f}%  acc={best_f1["accuracy"]*100:.2f}%')

        # Plot
        thrs = [r['threshold'] for r in rows]
        for metric, color in colors.items():
            vals = [r[metric] * 100 for r in rows]
            ax.plot(thrs, vals, color=color, lw=1.8, marker='o', ms=4, label=metric)
        ax.axvline(best_f1['threshold'], color='k', lw=1.5, ls='--',
                   label=f'best thr={best_f1["threshold"]:.2f}')
        ax.set_title(name.replace('_', ' ').upper(), fontsize=12)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Metric (%)')
        ax.legend(fontsize=9)
        ax.set_xlim(0.28, 0.97)
        ax.grid(alpha=0.3)

    fig.suptitle('Extension 2 – Optimal Decision Threshold Search\n'
                 '(Val set, all 4 models)', fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'fig_threshold_opt.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\nFigure saved: {fig_path}')

    json_path = os.path.join(OUT_DIR, 'threshold_opt.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved: {json_path}')

    # Print summary table
    print('\n=== Optimal Threshold Summary ===')
    print(f'{"Model":<12} {"Best thr":>8} {"F1@best":>10} {"F1@0.5":>10} {"ΔF1":>8}')
    for name, rows in results.items():
        best = max(rows, key=lambda r: r['f1'])
        at05 = next(r for r in rows if abs(r['threshold'] - 0.5) < 0.03)
        delta = (best['f1'] - at05['f1']) * 100
        print(f'{name:<12} {best["threshold"]:>8.2f} '
              f'{best["f1"]*100:>10.2f}% {at05["f1"]*100:>10.2f}% {delta:>+8.2f}pp')


if __name__ == '__main__':
    main()
