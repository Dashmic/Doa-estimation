"""
generate_raw.py
---------------
Generate raw complex signal snapshots for the DOA estimation dataset.

Signal model:
    x(t) = A(theta) @ s(t) + n(t)
    x(t) : (P,)   received signal at P sensors for snapshot t
    A     : (P,K)  steering matrix
    s(t)  : (K,)   source signals (i.i.d. complex Gaussian)
    n(t)  : (P,)   AWGN noise

Input tensor shape  : (2, P, T)   -- channel 0=real, channel 1=imag
Label tensor shape  : (121,)      -- multi-hot, DOA range -60..+60 deg, step 1

Usage:
    python datasets/generate_raw.py --T 16 --samples 2000000 --snr 10
                                    --out data/raw_t16_train.h5
"""

import argparse
import os
import time
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
from tqdm import tqdm

# Project-local import (run from project root)
from datasets.array_geometry import get_tca_positions, get_steering_matrix

# ── Constants ─────────────────────────────────────────────────────────────────
DOA_MIN   = -60          # degrees
DOA_MAX   =  60          # degrees
DOA_STEP  =   1          # degrees
DOA_GRID  = np.arange(DOA_MIN, DOA_MAX + DOA_STEP, DOA_STEP)  # 121 angles
NUM_CLASSES = len(DOA_GRID)    # 121
K_MIN     =  1           # min number of sources per sample
K_MAX     = 16           # max number of sources per sample
CHUNK_SIZE = 4096        # HDF5 chunk rows for efficient I/O


def angle_to_class(angle_deg: float) -> int:
    """Convert a DOA angle (deg) to its class index [0..120]."""
    return int(round((angle_deg - DOA_MIN) / DOA_STEP))


def simulate_snapshot(
    positions: np.ndarray,
    T: int,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple:
    """
    Simulate one data sample: K random sources, T snapshots.

    Returns
    -------
    x_real : np.ndarray (2, P, T)   real/imag parts of received signal
    label  : np.ndarray (121,)      multi-hot DOA label
    """
    P = len(positions)
    K = rng.integers(K_MIN, K_MAX + 1)

    # Draw K unique DOA angles from the grid
    chosen_idx = rng.choice(NUM_CLASSES, size=K, replace=False)
    thetas = DOA_GRID[chosen_idx]

    # Build steering matrix A: (P, K)
    A = get_steering_matrix(thetas, positions)

    # Source signals: i.i.d. CN(0, 1), shape (K, T)
    s = (rng.standard_normal((K, T)) + 1j * rng.standard_normal((K, T))) / np.sqrt(2)

    # Received signal without noise: (P, T)
    X_clean = A @ s

    # Signal power per sensor (average)
    signal_power = np.mean(np.abs(X_clean) ** 2)

    # Noise power from SNR definition: SNR = signal_power / noise_power
    snr_linear = 10 ** (snr_db / 10.0)
    noise_std   = np.sqrt(signal_power / snr_linear / 2)   # /2 for complex

    # AWGN noise: (P, T)
    noise = noise_std * (
        rng.standard_normal((P, T)) + 1j * rng.standard_normal((P, T))
    )

    X = X_clean + noise  # (P, T)

    # Stack real/imag -> (2, P, T)
    x_out = np.stack([X.real, X.imag], axis=0).astype(np.float32)

    # Multi-hot label
    label = np.zeros(NUM_CLASSES, dtype=np.float32)
    label[chosen_idx] = 1.0

    return x_out, label


def _worker(args):
    """
    Worker function for multiprocessing.
    Each worker generates a chunk of samples and writes to a temporary HDF5.
    """
    (worker_id, n_samples, T, snr_db, positions, save_path) = args

    P  = len(positions)
    rng = np.random.default_rng(seed=worker_id * 12345 + 7)

    X_buf = np.empty((n_samples, 2, P, T), dtype=np.float32)
    Y_buf = np.empty((n_samples, NUM_CLASSES), dtype=np.float32)

    for i in range(n_samples):
        X_buf[i], Y_buf[i] = simulate_snapshot(positions, T, snr_db, rng)

    chunk_n = min(CHUNK_SIZE, n_samples)
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('X', data=X_buf,
                         chunks=(chunk_n, 2, P, T), compression='lzf')
        f.create_dataset('Y', data=Y_buf,
                         chunks=(chunk_n, NUM_CLASSES), compression='lzf')

    return worker_id, n_samples


def merge_h5_files(part_paths: list, output_path: str, P: int, T: int) -> None:
    """Concatenate per-worker HDF5 files into one final file."""
    total = sum(
        h5py.File(p, 'r')['X'].shape[0] for p in part_paths
    )

    chunk_n = min(CHUNK_SIZE, total)
    with h5py.File(output_path, 'w') as fout:
        dset_x = fout.create_dataset(
            'X', shape=(total, 2, P, T), dtype='float32',
            chunks=(chunk_n, 2, P, T), compression='lzf'
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

    # Store metadata
    with h5py.File(output_path, 'a') as f:
        f.attrs['T']           = T
        f.attrs['num_sensors'] = P
        f.attrs['num_classes'] = NUM_CLASSES
        f.attrs['doa_min']     = DOA_MIN
        f.attrs['doa_max']     = DOA_MAX
        f.attrs['doa_step']    = DOA_STEP
        f.attrs['input_type']  = 'raw'


def generate(
    T: int,
    n_samples: int,
    snr_db: float,
    output_path: str,
    n_workers: int = None,
    M: int = 5,
    N: int = 6,
) -> None:
    """
    Top-level generation function.

    Parameters
    ----------
    T          : number of snapshots
    n_samples  : total samples to generate
    snr_db     : signal-to-noise ratio in dB
    output_path: path to final HDF5 file
    n_workers  : number of parallel processes (default: cpu_count - 1)
    M, N       : TCA parameters
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    positions = get_tca_positions(M, N)
    P = len(positions)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Split work across workers
    base  = n_samples // n_workers
    sizes = [base] * n_workers
    sizes[-1] += n_samples - sum(sizes)   # remainder to last worker

    tmp_dir = os.path.join(os.path.dirname(output_path), '_tmp_raw')
    os.makedirs(tmp_dir, exist_ok=True)

    args_list = [
        (wid, sizes[wid], T, snr_db, positions,
         os.path.join(tmp_dir, f'part_{wid}.h5'))
        for wid in range(n_workers)
    ]

    print(f'[generate_raw] T={T}, samples={n_samples}, SNR={snr_db}dB, '
          f'sensors={P}, workers={n_workers}')
    t0 = time.time()

    with Pool(n_workers) as pool:
        for wid, cnt in tqdm(
            pool.imap_unordered(_worker, args_list),
            total=n_workers, desc='Workers'
        ):
            pass  # progress handled by tqdm

    print(f'  Generation done in {time.time()-t0:.1f}s. Merging...')
    part_paths = [a[-1] for a in args_list]
    merge_h5_files(part_paths, output_path, P, T)

    # Clean up temp files
    for p in part_paths:
        os.remove(p)
    os.rmdir(tmp_dir)

    file_mb = os.path.getsize(output_path) / 1e6
    print(f'  Saved -> {output_path}  ({file_mb:.1f} MB)')


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate raw signal HDF5 dataset for DOA estimation'
    )
    parser.add_argument('--T',       type=int,   default=16,
                        help='Number of snapshots (default 16)')
    parser.add_argument('--samples', type=int,   default=2_000_000,
                        help='Total number of samples (default 2M)')
    parser.add_argument('--snr',     type=float, default=10.0,
                        help='Training SNR in dB (default 10)')
    parser.add_argument('--out',     type=str,   default=None,
                        help='Output HDF5 path (auto-named if omitted)')
    parser.add_argument('--workers', type=int,   default=None,
                        help='Number of parallel workers')
    parser.add_argument('--M',       type=int,   default=5)
    parser.add_argument('--N',       type=int,   default=6)
    args = parser.parse_args()

    if args.out is None:
        args.out = f'data/raw_t{args.T}_n{args.samples//1_000_000}M.h5'

    generate(
        T=args.T,
        n_samples=args.samples,
        snr_db=args.snr,
        output_path=args.out,
        n_workers=args.workers,
        M=args.M,
        N=args.N,
    )
