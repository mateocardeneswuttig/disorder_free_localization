
"""
Replacement for the MATLAB expv.m routine.

The original expv.m implements a Krylov/Arnoldi scheme to compute w = exp(t*A) v.

In Python, we rely on SciPy's expm_multiply which is robust and efficient for sparse matrices.

We keep the same return signature (w, err, hump) to ease porting.
"""
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def expv(t: complex, A, v: np.ndarray, tol: float = 1e-7, m: Optional[int] = None) -> Tuple[np.ndarray, float, float]:
    """
    Compute w = exp(t*A) @ v.

    Parameters
    ----------
    t : complex or float
        Time step (can be complex, e.g. -1j*dt).
    A : (n,n) array_like or sparse matrix
        Generator matrix.
    v : (n,) array_like
        Vector.
    tol : float
        Tolerance hint (used as atol/rtol).
    m : int, optional
        Krylov subspace dimension in MATLAB version (ignored; SciPy chooses internally).

    Returns
    -------
    w : ndarray
    err : float
        A crude error indicator (0.0 here; SciPy does not expose the same metric).
    hump : float
        A crude "hump" indicator (||w|| / ||v||).
    """
    v = np.asarray(v)
    # SciPy supports linear operator / sparse
    w = spla.expm_multiply(t * A, v, atol=tol, rtol=tol)
    vnorm = np.linalg.norm(v)
    hump = float(np.linalg.norm(w) / vnorm) if vnorm != 0 else float("nan")
    err = 0.0
    return w, err, hump
