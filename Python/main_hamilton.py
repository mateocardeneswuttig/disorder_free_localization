
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .operators import build_local_operators
from .basis import fock2label
from .expv import expv


def run_main_hamilton(
    J: float = 1.0,
    omega: float = np.pi/2,
    g: float = 0.25,
    alpha: complex = np.sqrt(2),
    L: int = 4,
    filling: int | None = None,
    Nmax: int = 1,
    times: np.ndarray | None = None,
    verbose: bool = True,
):
    """
    Python port of main_Hamilton.m.

    Notes
    -----
    * This is a direct, readability-first port. For large L it will be slow because it loops over basis states.
    * The MATLAB script contains plotting; here we return arrays so you can plot with matplotlib.

    Returns
    -------
    results : dict
        Contains H, psi_t (list), matter_expect (array), phonon_expect (array), times
    """
    if filling is None:
        filling = L  # half-filling in the original script
    if times is None:
        times = np.arange(0, 200 + 1e-12, 2.0)

    # Build local operators
    ops = build_local_operators(L=L, filling=filling, Nmax=Nmax, verbose=verbose)

    dim = ops.n_up[0].shape[0]
    I = sp.identity(dim, format="csr", dtype=np.complex128)

    # Build Hamiltonian
    H_e = sp.csr_matrix((dim, dim), dtype=np.complex128)
    H_ph = sp.csr_matrix((dim, dim), dtype=np.complex128)
    H_eph = sp.csr_matrix((dim, dim), dtype=np.complex128)
    Matter = sp.csr_matrix((dim, dim), dtype=np.complex128)
    Ph = sp.csr_matrix((dim, dim), dtype=np.complex128)

    for m in range(L):
        H_e = H_e - J * ops.tunneling_up[m] - J * ops.tunneling_down[m]
        H_ph = H_ph + omega * (ops.phonon_creator[m] @ ops.phonon_annihilator[m])
        n_i = ops.n_up[m] + ops.n_down[m]
        H_eph = H_eph + g * (n_i - I) @ (ops.phonon_annihilator[m] + ops.phonon_creator[m]) @ (ops.phonon_annihilator[m] + ops.phonon_creator[m])
        Matter = Matter + ops.n_up[m]
        Ph = Ph + ops.n_ph[m]

    H = H_e + H_ph + H_eph

    if verbose:
        # A few eigenvalues for diagnostics (as in MATLAB eigs)
        try:
            evals = spla.eigs(H, k=min(6, dim-2), which="SR", return_eigenvectors=False)
            print("Lowest eigenvalues (approx):", np.sort(evals.real))
        except Exception as e:
            print("eigs failed:", e)

    # --- Initial state construction (ported from the MATLAB script) ---
    # The MATLAB code uses 12-entry vectors (3*L) because L=4 there. We generalize only for L=4 here.
    if L != 4:
        raise NotImplementedError("The provided initial-state superposition is hard-coded for L=4 (as in the MATLAB script).")

    up_states = [
        np.array([1,0,0,1,0,0,0,0,0,0,0,0], dtype=int),
        np.array([1,0,0,0,0,0,1,0,0,0,0,0], dtype=int),
        np.array([1,0,0,0,0,0,0,0,0,1,0,0], dtype=int),
        np.array([0,0,0,1,0,0,1,0,0,0,0,0], dtype=int),
        np.array([0,0,0,1,0,0,0,0,0,1,0,0], dtype=int),
        np.array([0,0,0,0,0,0,1,0,0,1,0,0], dtype=int),
    ]
    up_coeffs = np.array([1-1j, 2, 1+1j, 1+1j, 2j, 1j-1], dtype=np.complex128)

    down_states = [
        np.array([0,1,0,0,1,0,0,0,0,0,0,0], dtype=int),
        np.array([0,1,0,0,0,0,0,1,0,0,0,0], dtype=int),
        np.array([0,1,0,0,0,0,0,0,0,0,1,0], dtype=int),
        np.array([0,0,0,0,1,0,0,1,0,0,0,0], dtype=int),
        np.array([0,0,0,0,1,0,0,0,0,0,1,0], dtype=int),
        np.array([0,0,0,0,0,0,0,1,0,0,1,0], dtype=int),
    ]
    down_coeffs = np.array([1-1j, 2, 1+1j, 1+1j, 2j, 1j-1], dtype=np.complex128)

    phononmax = np.array([0,0,Nmax,0,0,Nmax,0,0,Nmax,0,0,Nmax], dtype=int)

    psi0 = np.zeros(dim, dtype=np.complex128)

    for k in range(6):
        for n in range(6):
            statefirst = up_states[k] + down_states[n]
            statelast = statefirst + phononmax

            begin_ind = fock2label(statefirst, L, filling, Nmax)
            end_ind = fock2label(statelast, L, filling, Nmax)

            # sum over all phonon configurations between begin_ind and end_ind (as MATLAB does)
            for cnt in range(begin_ind, end_ind + 1):
                # In MATLAB: coeff = exp(-L*(abs(alpha)^2)/2) * alpha^(sum phonons)/sqrt(prod factorials)
                # We re-create that from the decoded phonon occupations of this basis state.
                # To avoid calling label2fock repeatedly, we approximate by using the fact that labels in the range
                # differ only by phonons; however to be safe we decode.
                # (This is still small for L=4, Nmax<=1.)
                from .basis import label2fock
                psi = label2fock(cnt, L, filling, Nmax)
                ph = psi[2::3]
                coeff = np.exp(-L*(abs(alpha)**2)/2.0)
                coeff *= (alpha ** int(ph.sum()))
                # factorial term
                import math
                denom = 1.0
                for x in ph:
                    denom *= math.factorial(int(x))
                coeff /= np.sqrt(denom)

                amp = up_coeffs[k] * down_coeffs[n] * coeff
                psi0[cnt - 1] += amp

    # normalize
    norm0 = np.linalg.norm(psi0)
    if norm0 != 0:
        psi0 = psi0 / norm0

    # --- Time evolution ---
    psi = psi0.copy()
    psi_t = [psi.copy()]
    matter_expect = [np.real(np.vdot(psi, Matter @ psi))]
    phonon_expect = [np.real(np.vdot(psi, Ph @ psi))]

    for t_prev, t_next in zip(times[:-1], times[1:]):
        dt = t_next - t_prev
        psi, _, _ = expv(-1j * dt, H, psi, tol=1e-9, m=min(dim, 30))
        # renormalize to avoid drift
        psi = psi / np.linalg.norm(psi)

        psi_t.append(psi.copy())
        matter_expect.append(np.real(np.vdot(psi, Matter @ psi)))
        phonon_expect.append(np.real(np.vdot(psi, Ph @ psi)))

    return {
        "H": H,
        "psi_t": psi_t,
        "times": np.array(times, dtype=float),
        "matter_expect": np.array(matter_expect),
        "phonon_expect": np.array(phonon_expect),
    }
