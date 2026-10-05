"""
generate_cov.py
---------------
Generate sample covariance matrix dataset for DOA estimation.

Covariance estimate (MLE):
    R_hat = (1/T) * X @ X^H      where X: (P, T) complex received signal

The 2-channel input stored in HDF5:
    channel 0 : real(R_hat)    shape (P, P) = (12, 12)
    channel 1 : imag(R_hat)    shape (P, P)
    -> stored as (2, P, P) float32

Label tensor shape : (121,)  multi-hot, identical definition to generate_raw.py

Usage:
    python datasets/generate_cov.py --T 16 --samples 2000000 --snr 10
                                    --out data/cov_t16_train.h5
"""

import argparse
import os
import time
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
from tqdm import tqdm

from datasets.array_geometry import get_tca_positions, get_steering_matrix

# ── Constants ─────────────────────────────────────────────────────────────────
DOA_MIN     = -60
DOA_MAX     =  60
DOA_STEP    =   1
DOA_GRID    = np.arange(DOA_MIN, DOA_MAX + DOA_STEP, DOA_STEP)  # 121 angles
NUM_CLASSES = len(DOA_GRID)
K_MIN       =  1
K_MAX       = 16
CHUNK_SIZE  = 4096


def angle_to_class(angle_deg: float) -> int:
    return int(round((angle_deg - DOA_MIN) / DOA_STEP))


def simulate_cov(
    positions: np.ndarray,
    T: int,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple:
    """
    Simulate one sample covariance matrix with K random sources.

    Returns
    -------
    cov_2ch : np.ndarray (2, P, P) float32
        Channel 0 = real(R_hat), Channel 1 = imag(R_hat).
    label   : np.ndarray (121,) float32
        Multi-hot label.
    """
    P = len(positions)
    K = rng.integers(K_MIN, K_MAX + 1)

    # Random unique DOA indices
    chosen_idx = rng.choice(NUM_CLASSES, size=K, replace=False)
    thetas = DOA_GRID[chosen_idx]

    # Steering matrix (P, K)
    A = get_steering_matrix(thetas, positions)

    # Source signals CN(0,1): (K, T)
    s = (rng.standard_normal((K, T)) + 1j * rng.standard_normal((K, T))) / np.sqrt(2)
    X_clean = A @ s      # (P, T)

    # Noise
    signal_power = np.mean(np.abs(X_clean) ** 2)
    snr_linear   = 10 ** (snr_db / 10.0)
    noise_std    = np.sqrt(signal_power / snr_linear / 2)
    noise = noise_std * (
        rng.standard_normal((P, T)) + 1j * rng.standard_normal((P, T))
    )

    X = X_clean + noise  # (P, T)

    # Sample covariance estimate: R_hat = X @ X^H / T
    R_hat = (X @ X.conj().T) / T  # (P, P) Hermitian

    # 2-channel representation
    cov_2ch = np.stack([R_hat.real, R_hat.imag], axis=0).astype(np.float32)

    # Multi-hot label
    label = np.zeros(NUM_CLASSES, dtype=np.float32)
    label[chosen_idx] = 1.0

    return cov_2ch, label


def _worker(args):
    (worker_id, n_samples, T, snr_db, positions, save_path) = args

    P   = len(positions)
    rng = np.random.default_rng(seed=worker_id * 99991 + 13)

    X_buf = np.empty((n_samples, 2, P, P), dtype=np.float32)
    Y_buf = np.empty((n_samples, NUM_CLASSES), dtype=np.float32)

    for i in range(n_samples):
        X_buf[i], Y_buf[i] = simulate_cov(positions, T, snr_db, rng)

    chunk_n = min(CHUNK_SIZE, n_samples)
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('X', data=X_buf,
                         chunks=(chunk_n, 2, P, P), compression='lzf')
        f.create_dataset('Y', data=Y_buf,
                         chunks=(chunk_n, NUM_CLASSES), compression='lzf')

    return worker_id, n_samples


def merge_h5_files(part_paths: list, output_path: str, P: int, T: int) -> None:
    total = sum(h5py.File(p, 'r')['X'].shape[0] for p in part_paths)

    chunk_n = min(CHUNK_SIZE, total)
    with h5py.File(output_path, 'w') as fout:
        dset_x = fout.create_dataset(
            'X', shape=(total, 2, P, P), dtype='float32',
            chunks=(chunk_n, 2, P, P), compression='lzf'
        )
        dset_y = fout.create_dataset(
            'Y', shape=(total, NUM_CLASSES), dtype='float32',
            chunks=(chunk_n, NUM_CLASSES), compression='lzf'
        )
        cursor = 0
        for p in part_paths:
            with h5py.File(p, 'r') as fin:
                n = fin['X'].shape[0]
                dset_x[cursor:cursor + n] = fin['X'][:]
                dset_y[cursor:cursor + n] = fin['Y'][:]
                cursor += n

    with h5py.File(output_path, 'a') as f:
        f.attrs['T']           = T
        f.attrs['num_sensors'] = P
        f.attrs['num_classes'] = NUM_CLASSES
        f.attrs['doa_min']     = DOA_MIN
        f.attrs['doa_max']     = DOA_MAX
        f.attrs['doa_step']    = DOA_STEP
        f.attrs['input_type']  = 'cov'


def generate(
    T: int,
    n_samples: int,
    snr_db: float,
    output_path: str,
    n_workers: int = None,
    M: int = 5,
    N: int = 6,
) -> None:
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    positions = get_tca_positions(M, N)
    P = len(positions)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    base  = n_samples // n_workers
    sizes = [base] * n_workers
    sizes[-1] += n_samples - sum(sizes)

    tmp_dir = os.path.join(os.path.dirname(output_path), '_tmp_cov')
    os.makedirs(tmp_dir, exist_ok=True)

    args_list = [
        (wid, sizes[wid], T, snr_db, positions,
         os.path.join(tmp_dir, f'part_{wid}.h5'))
        for wid in range(n_workers)
    ]

    print(f'[generate_cov] T={T}, samples={n_samples}, SNR={snr_db}dB, '
          f'sensors={P}, workers={n_workers}')
    t0 = time.time()

    with Pool(n_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_worker, args_list),
            total=n_workers, desc='Workers'
        ):
            pass

    print(f'  Generation done in {time.time()-t0:.1f}s. Merging...')
    part_paths = [a[-1] for a in args_list]
    merge_h5_files(part_paths, output_path, P, T)

    for p in part_paths:
        os.remove(p)
    os.rmdir(tmp_dir)

    file_mb = os.path.getsize(output_path) / 1e6
    print(f'  Saved -> {output_path}  ({file_mb:.1f} MB)')


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate covariance matrix HDF5 dataset'
    )
    parser.add_argument('--T',       type=int,   default=16)
    parser.add_argument('--samples', type=int,   default=2_000_000)
    parser.add_argument('--snr',     type=float, default=10.0)
    parser.add_argument('--out',     type=str,   default=None)
    parser.add_argument('--workers', type=int,   default=None)
    parser.add_argument('--M',       type=int,   default=5)
    parser.add_argument('--N',       type=int,   default=6)
    args = parser.parse_args()

    if args.out is None:
        args.out = f'data/cov_t{args.T}_n{args.samples//1_000_000}M.h5'

    generate(
        T=args.T,
        n_samples=args.samples,
        snr_db=args.snr,
        output_path=args.out,
        n_workers=args.workers,
        M=args.M,
        N=args.N,
    )
