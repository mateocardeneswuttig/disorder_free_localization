
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .operators import build_local_operators
from .basis import fock2label, label2fock
from .expv import expv


def run_main_heffect(
    J: float = 1.0,
    omega: float = np.pi/2,
    g: float = 0.25,
    alpha: float = np.sqrt(2),
    L: int = 4,
    filling: int | None = None,
    Nmax: int = 1,
    times: np.ndarray | None = None,
    verbose: bool = True,
):
    """
    Python port of main_Heffect.m (effective model).

    Returns a dict with Hamiltonian, initial state, imbalance vs time, etc.
    """
    if filling is None:
        filling = L
    if times is None:
        times = np.arange(0, 1000 + 1e-12, 10.0)

    ops = build_local_operators(L=L, filling=filling, Nmax=Nmax, verbose=verbose)
    dim = ops.n_up[0].shape[0]
    I = sp.identity(dim, format="csr", dtype=np.complex128)

    # Effective parameters from MATLAB
    J_star = J * np.exp(-0.5 * (g / omega) ** 2 * (alpha ** 2 + 2 * alpha + 1))
    omega_star = omega - (g ** 2) / omega

    # Build Hamiltonian (port of MATLAB loop using elementwise multiplications for diagonal pieces)
    H = sp.csr_matrix((dim, dim), dtype=np.complex128)
    imbOp = sp.csr_matrix((dim, dim), dtype=np.complex128)

    for m in range(L):
        nup = ops.n_up[m]
        ndn = ops.n_down[m]
        nph = ops.n_ph[m]

        H = H - J_star * ops.tunneling_up[m] - J_star * ops.tunneling_down[m]
        H = H + omega_star * nph

        # diagonal combinations; use elementwise product to mimic MATLAB ".*"
        n_tot_minus_1 = (nup + ndn - I)
        H = H + 2 * g * n_tot_minus_1.multiply(nph + 0.5 * I)

        H = H - 4 * (g ** 2) / omega * (nup - 0.5 * I).multiply(ndn - 0.5 * I).multiply(nph + 0.5 * I)

    if verbose:
        try:
            evals = spla.eigs(H, k=min(6, dim-2), which="SR", return_eigenvectors=False)
            print("Lowest eigenvalues (approx):", np.sort(evals.real))
        except Exception as e:
            print("eigs failed:", e)

    # Initial coherent(-like) state in the alternating matter configuration (as MATLAB)
    if L != 4:
        raise NotImplementedError("The provided initial state follows the MATLAB script; generalized L is not implemented here.")

    target = np.mod(np.arange(1, L + 1), 2)  # [1,0,1,0] for L=4
    first = np.zeros(3 * L, dtype=int)
    first[0::3] = target
    first[1::3] = target

    last = Nmax * np.ones(3 * L, dtype=int)
    last[0::3] = target
    last[1::3] = target

    begin_ind = fock2label(first, L, filling, Nmax)
    end_ind = fock2label(last, L, filling, Nmax)

    psi0 = np.zeros(dim, dtype=np.complex128)

    for cnt in range(begin_ind, end_ind + 1):
        psi = label2fock(cnt, L, filling, Nmax)
        ph = psi[2::3]

        coeff = np.exp(-L * (abs(alpha) ** 2) / 2.0) * (alpha ** int(ph.sum()))
        import math
        denom = 1.0
        for x in ph:
            denom *= math.factorial(int(x))
        coeff /= np.sqrt(denom)

        psi0[cnt - 1] = coeff

    psi0 = psi0 / np.linalg.norm(psi0)

    # Build imbOp from MATLAB definition (it constructs using psi0 and n_up operators)
    # MATLAB: imbOp = sum_m (2 * psi0' * n_up{m} * psi0 - 1) * n_up{m} / L
    for m in range(L):
        nup = ops.n_up[m]
        coeff_m = 2 * np.vdot(psi0, (nup @ psi0)) - 1
        imbOp = imbOp + (coeff_m / L) * nup

    # Time evolution and imbalance
    psi = psi0.copy()
    imb = [float(np.real(np.vdot(psi, imbOp @ psi)))]

    for t_prev, t_next in zip(times[:-1], times[1:]):
        dt = t_next - t_prev
        psi, _, _ = expv(-1j * dt, H, psi, tol=1e-9, m=min(dim, 30))
        psi = psi / np.linalg.norm(psi)
        imb.append(float(np.real(np.vdot(psi, imbOp @ psi))))

    return {
        "H": H,
        "psi0": psi0,
        "times": np.array(times, dtype=float),
        "imbalance": np.array(imb, dtype=float),
        "J_star": float(J_star),
        "omega_star": float(omega_star),
    }
