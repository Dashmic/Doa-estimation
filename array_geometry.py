"""
array_geometry.py
-----------------
Thinned Coprime Array (TCA) geometry definition.

Strictly follows paper Eq.(3) and Fig.2  (M=5, N=6):

  X1 = { n*M*d | 0 <= n <= N-1 }         -> {0,5,10,15,20,25}   (N=6 elements)
  X2 = { m*N*d | 1 <= m <= floor(M/2) }  -> {6,12}               (floor(5/2)=2 elements)
  X3 = { (m+M+1)*N*d | 0 <= m <= M-2 }  -> {36,42,48,54}         (M-1=4 elements)

  TCA = X1 ∪ X2 ∪ X3
      = {0, 5, 6, 10, 12, 15, 20, 25, 36, 42, 48, 54}  -> 12 sensors

  Total sensors S = M + N + floor(M/2) - 1 = 5+6+2-1 = 12  (Eq.4 verified)

Reference:
  "A Unified Approach for Target Direction Finding Based on CNNs"
  IEEE MLSP 2020, DOI: 10.1109/MLSP49062.2020.9231787
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def get_tca_positions(M: int = 5, N: int = 6) -> np.ndarray:
    """
    Compute TCA sensor positions per paper Eq.(3).

    Three sub-arrays:
      X1 = { n*M | 0 <= n <= N-1 }              (N elements)
      X2 = { m*N | 1 <= m <= floor(M/2) }       (floor(M/2) elements, 0 excluded)
      X3 = { (m+M+1)*N | 0 <= m <= M-2 }        (M-1 elements)

    For M=5, N=6:
      X1 = {0,5,10,15,20,25}
      X2 = {6,12}
      X3 = {36,42,48,54}
      TCA = {0,5,6,10,12,15,20,25,36,42,48,54}  -> 12 sensors

    Total: S = M + N + floor(M/2) - 1  (paper Eq.4)

    Parameters
    ----------
    M : int   First coprime parameter (default 5)
    N : int   Second coprime parameter (default 6)

    Returns
    -------
    positions : np.ndarray, shape (S,)  sorted unique positions (units of d=λ/2)
    """
    assert np.gcd(M, N) == 1, f"M={M} and N={N} must be coprime"

    X1 = [n * M for n in range(N)]                       # {0,5,10,15,20,25}
    X2 = [m * N for m in range(1, M // 2 + 1)]           # {6,12}
    X3 = [(m + M + 1) * N for m in range(M - 1)]         # {36,42,48,54}

    positions = sorted(set(X1 + X2 + X3))
    expected  = M + N + M // 2 - 1
    assert len(positions) == expected, \
        f"Expected {expected} sensors (Eq.4), got {len(positions)}: {positions}"
    return np.array(positions, dtype=np.float64)



def get_steering_vector(theta_deg: float, positions: np.ndarray) -> np.ndarray:
    """
    Compute the steering vector a(theta) for a given DOA angle.

    Assumes inter-element spacing d = lambda/2, so the phase shift per unit
    position is pi * sin(theta).

    Parameters
    ----------
    theta_deg : float
        Direction of arrival in degrees.
    positions : np.ndarray, shape (P,)
        Array sensor positions in units of d = lambda/2.

    Returns
    -------
    a : np.ndarray, shape (P,) complex128
        Steering vector.
    """
    theta_rad = np.deg2rad(theta_deg)
    # Phase: 2*pi*(d/lambda)*sin(theta)*pos = pi*sin(theta)*pos  (d=lambda/2)
    phase = np.pi * np.sin(theta_rad) * positions
    return np.exp(1j * phase)


def get_steering_matrix(thetas_deg: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Compute the steering matrix A for multiple DOA angles.

    Parameters
    ----------
    thetas_deg : np.ndarray, shape (K,)
        Array of DOA angles in degrees.
    positions : np.ndarray, shape (P,)
        Sensor positions.

    Returns
    -------
    A : np.ndarray, shape (P, K) complex128
        Steering matrix; each column is a(theta_k).
    """
    A = np.stack(
        [get_steering_vector(th, positions) for th in thetas_deg], axis=-1
    )
    return A  # (P, K)


def plot_array(positions: np.ndarray, M: int = 5, N: int = 6,
               save_path: 'str | None' = None) -> None:
    """
    Visualise TCA sensor layout along the aperture axis.

    Parameters
    ----------
    positions : np.ndarray
        Sensor positions.
    M, N : int
        TCA parameters used for the title.
    save_path : str or None
        If provided, save the figure to this path.
    """
    sub1 = set(k * N for k in range(M))
    sub2 = set(k * M for k in range(N + 2))

    fig, ax = plt.subplots(figsize=(10, 2))
    for p in positions:
        color = 'royalblue' if p in sub1 and p in sub2 else \
                ('royalblue' if p in sub1 else 'tomato')
        ax.plot(p, 0, 'o', color=color, markersize=10, markeredgecolor='k')
    ax.set_xlim(-2, max(positions) + 2)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Position (d = λ/2 units)')
    ax.set_title(
        f'Thinned Coprime Array  M={M}, N={N}  |  {len(positions)} sensors\n'
        f'Blue=Sub-array 1  Red=Sub-array 2'
    )
    ax.yaxis.set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ── quick sanity check ────────────────────────────────────────────────────────
if __name__ == '__main__':
    pos = get_tca_positions(M=5, N=6)
    print(f'TCA positions (M=5, N=6): {pos.tolist()}')
    print(f'Number of sensors       : {len(pos)}')
    plot_array(pos)
