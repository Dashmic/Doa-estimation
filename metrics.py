"""
metrics.py
----------
Evaluation metrics for multi-label DOA estimation.

Definitions (element-wise across all classes, following the paper):
  Let y_pred (after threshold) and y_true be binary (0/1) vectors.

  TP = sum(y_pred * y_true)
  FP = sum(y_pred * (1 - y_true))
  TN = sum((1-y_pred) * (1-y_true))
  FN = sum((1-y_pred) * y_true)

  Accuracy    = (TP + TN) / (TP + FP + TN + FN)
  Precision   = TP / (TP + FP)          (per-class macro average)
  Recall      = TP / (TP + FN)          (per-class macro average)
  Specificity = TN / (TN + FP)          (per-class macro average)

Usage:
    python eval/metrics.py --config configs/raw_t16.yaml
                           --checkpoint results/checkpoints/raw_t16_best.pth
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from datasets.data_loader import get_dataloaders
from datasets.array_geometry import get_tca_positions, get_steering_matrix
from models.resnext_doa import ResNeXtDOA

# DOA configuration
DOA_MIN  = -60
DOA_MAX  =  60
DOA_STEP =   1
DOA_GRID = np.arange(DOA_MIN, DOA_MAX + DOA_STEP, DOA_STEP)


# ── Core metric computation ────────────────────────────────────────────────────

def compute_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute Accuracy, Precision, Recall, Specificity.

    Parameters
    ----------
    y_pred : np.ndarray (N, C)  raw sigmoid probabilities
    y_true : np.ndarray (N, C)  binary ground truth
    threshold : float           decision threshold

    Returns
    -------
    dict with keys: accuracy, precision, recall, specificity
    """
    pred_bin = (y_pred >= threshold).astype(np.float32)
    gt       = y_true.astype(np.float32)

    TP = (pred_bin * gt).sum(axis=0)        # (C,)
    FP = (pred_bin * (1 - gt)).sum(axis=0)
    TN = ((1 - pred_bin) * (1 - gt)).sum(axis=0)
    FN = ((1 - pred_bin) * gt).sum(axis=0)

    eps = 1e-8
    accuracy    = (TP + TN) / (TP + FP + TN + FN + eps)
    precision   = TP / (TP + FP + eps)
    recall      = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)

    return {
        'accuracy'   : float(accuracy.mean()),
        'precision'  : float(precision.mean()),
        'recall'     : float(recall.mean()),
        'specificity': float(specificity.mean()),
        # Per-class arrays (for per-class analysis)
        '_accuracy_per_class'   : accuracy.tolist(),
        '_precision_per_class'  : precision.tolist(),
        '_recall_per_class'     : recall.tolist(),
        '_specificity_per_class': specificity.tolist(),
    }


# ── Inference on test set ──────────────────────────────────────────────────────

def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple:
    """
    Run inference on a DataLoader and collect predictions + ground truth.

    Returns
    -------
    y_pred : np.ndarray (N, 121)  sigmoid probabilities
    y_true : np.ndarray (N, 121)  binary labels
    """
    model.eval()
    preds_list  = []
    labels_list = []

    with torch.no_grad():
        for X, Y in tqdm(loader, desc='Evaluating'):
            X = X.to(device, non_blocking=True)
            out = model(X).cpu().numpy()
            preds_list.append(out)
            labels_list.append(Y.numpy())

    y_pred = np.concatenate(preds_list,  axis=0)
    y_true = np.concatenate(labels_list, axis=0)
    return y_pred, y_true


# ── SNR sweep evaluation ───────────────────────────────────────────────────────

def evaluate_snr_sweep(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    snr_list: list = None,
) -> pd.DataFrame:
    """
    Evaluate model at multiple SNR levels by generating fresh test data on-the-fly.
    This avoids needing pre-generated files for each SNR — uses in-memory batches.

    Returns a DataFrame with columns: snr, accuracy, precision, recall, specificity
    """
    from datasets.generate_raw import simulate_snapshot
    from datasets.generate_cov  import simulate_cov

    if snr_list is None:
        snr_list = list(range(0, 22, 2))   # 0, 2, 4, ..., 20 dB

    positions  = get_tca_positions(cfg.get('M', 5), cfg.get('N', 6))
    itype      = cfg['input_type']
    T          = cfg['T']
    n_per_snr  = 10_000   # samples per SNR point (fast but representative)
    rng        = np.random.default_rng(42)

    rows = []
    model.eval()

    for snr in snr_list:
        X_list, Y_list = [], []
        for _ in range(n_per_snr):
            if itype == 'raw':
                x, y = simulate_snapshot(positions, T, snr, rng)
            else:
                x, y = simulate_cov(positions, T, snr, rng)
            X_list.append(x)
            Y_list.append(y)

        X_arr = torch.from_numpy(np.stack(X_list, axis=0))   # (N, 2, P, T/P)
        Y_arr = np.stack(Y_list, axis=0)

        # Batch inference
        preds = []
        bs = 512
        with torch.no_grad():
            for i in range(0, len(X_arr), bs):
                batch = X_arr[i:i+bs].to(device)
                preds.append(model(batch).cpu().numpy())
        y_pred = np.concatenate(preds, axis=0)

        m = compute_metrics(y_pred, Y_arr)
        rows.append({
            'snr'        : snr,
            'accuracy'   : m['accuracy']    * 100,
            'precision'  : m['precision']   * 100,
            'recall'     : m['recall']      * 100,
            'specificity': m['specificity'] * 100,
        })
        print(f"  SNR={snr:3d}dB  acc={m['accuracy']*100:.2f}%  "
              f"prec={m['precision']*100:.2f}%  rec={m['recall']*100:.2f}%  "
              f"spec={m['specificity']*100:.2f}%")

    return pd.DataFrame(rows)


# ── Table generation (paper Table 2-5) ────────────────────────────────────────

def generate_tables(results_dir: str = 'results') -> None:
    """
    Load all 4 model results and generate Tables 2-5 in CSV + LaTeX.

    Expects JSON files: results/{raw_t16,raw_t32,cov_t16,cov_t32}_metrics.json
    """
    configs = ['raw_t16', 'raw_t32', 'cov_t16', 'cov_t32']
    tables_dir = os.path.join(results_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    # --- Table 2-5: one table per model ---
    for cfg_name in configs:
        json_path = os.path.join(results_dir, f'{cfg_name}_metrics.json')
        if not os.path.isfile(json_path):
            print(f'  [skip] {json_path} not found')
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data['snr_results'])
        df = df[['snr', 'accuracy', 'precision', 'recall', 'specificity']]
        df.columns = ['SNR (dB)', 'Accuracy (%)', 'Precision (%)',
                      'Recall (%)', 'Specificity (%)']
        df = df.round(2)

        # CSV
        csv_path = os.path.join(tables_dir, f'table_{cfg_name}.csv')
        df.to_csv(csv_path, index=False)

        # LaTeX
        tex_path = os.path.join(tables_dir, f'table_{cfg_name}.tex')
        itype = 'Raw' if 'raw' in cfg_name else 'Cov'
        T_val = cfg_name.split('t')[-1]
        caption = (f'Performance of {itype} signal input with T={T_val} snapshots '
                   f'across different SNR levels.')
        latex = df.to_latex(index=False, caption=caption,
                            label=f'tab:{cfg_name}', float_format='%.2f')
        with open(tex_path, 'w') as f:
            f.write(latex)

        print(f'  Saved: {csv_path}  {tex_path}')

    print('All tables generated.')


# ── Main evaluation script ─────────────────────────────────────────────────────

def run_eval(cfg: dict, checkpoint_path: str, results_dir: str = 'results') -> None:
    """
    Evaluate one model: compute metrics on test set + SNR sweep.
    Saves JSON with all results.
    """
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    itype    = cfg['input_type']
    run_name = f"{itype}_t{cfg['T']}"

    # Load model
    model = ResNeXtDOA(
        num_classes=cfg['num_classes'],
        input_type=itype,
        pretrained=False,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f'[metrics] Loaded {checkpoint_path}')

    # Test set evaluation (at training SNR)
    loaders = get_dataloaders(
        train_h5=cfg['train_h5'], val_h5=cfg['val_h5'], test_h5=cfg['test_h5'],
        input_type=itype, batch_size=256, num_workers=cfg['num_workers'],
        pin_memory=(device.type == 'cuda'),
    )
    print(f'Evaluating on test set (SNR={cfg["snr_train"]}dB)...')
    y_pred, y_true = evaluate_model(model, loaders['test'], device)
    test_metrics   = compute_metrics(y_pred, y_true)
    print(f"  acc={test_metrics['accuracy']*100:.2f}%  "
          f"prec={test_metrics['precision']*100:.2f}%  "
          f"rec={test_metrics['recall']*100:.2f}%  "
          f"spec={test_metrics['specificity']*100:.2f}%")

    # SNR sweep
    print('Running SNR sweep (0-20dB)...')
    snr_df   = evaluate_snr_sweep(model, cfg, device,
                                   snr_list=cfg.get('snr_eval_range', list(range(0, 22, 2))))
    snr_rows = snr_df.to_dict(orient='records')

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    out = {
        'run_name'   : run_name,
        'test_metrics': test_metrics,
        'snr_results' : snr_rows,
        'history'    : ckpt.get('history', {}),
    }
    json_path = os.path.join(results_dir, f'{run_name}_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Results saved: {json_path}')


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate DOA model and generate tables')
    parser.add_argument('--config',     type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--results',    type=str, default='results')
    parser.add_argument('--all',        action='store_true',
                        help='Generate all tables from existing JSON results')
    args = parser.parse_args()

    if args.all:
        generate_tables(args.results)
    else:
        if args.checkpoint is None:
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            itype = cfg['input_type']
            T     = cfg['T']
            args.checkpoint = os.path.join(cfg['checkpoint_dir'],
                                           f'{itype}_t{T}_best.pth')
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        run_eval(cfg, args.checkpoint, args.results)
