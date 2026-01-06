
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import scipy.sparse as sp

from .basis import label2fock, fock2label


@dataclass
class LocalOperators:
    tunneling_up: List[sp.csr_matrix]
    tunneling_down: List[sp.csr_matrix]
    phonon_creator: List[sp.csr_matrix]
    phonon_annihilator: List[sp.csr_matrix]
    n_up: List[sp.csr_matrix]
    n_down: List[sp.csr_matrix]
    n_ph: List[sp.csr_matrix]


def build_local_operators(L: int, filling: int, Nmax: int, verbose: bool = True) -> LocalOperators:
    """
    Port of the operator construction loops in main_Hamilton.m / main_Heffect.m.

    Builds, for each site m=1..L (1-based in the MATLAB code), the sparse operators:
      - tunneling_up[m], tunneling_down[m]
      - phonon_creator[m], phonon_annihilator[m]
      - n_up[m], n_down[m], n_ph[m]

    Basis dimension:
      dim = C(2L, filling) * (Nmax+1)^L
    """
    dim = int(sp.special.comb(2*L, filling, exact=True)) * int((Nmax + 1) ** L) if hasattr(sp, "special") else None
    # avoid depending on scipy.special here:
    from math import comb
    dim = comb(2*L, filling) * (Nmax + 1) ** L

    I = sp.identity(dim, format="csr", dtype=np.complex128)

    tunn_up = []
    tunn_dn = []
    bdag = []
    b = []
    nup = []
    ndn = []
    nph = []

    for m in range(1, L + 1):
        if verbose:
            print(f"Building operators for site m={m}/{L}")

        # vectors as in MATLAB
        ladder_vec = np.zeros(3 * L, dtype=int)
        ladder_vec[3 * m - 1] = 1  # 1-based -> index 3*m

        vec_up = np.zeros(3 * L, dtype=int)
        vec_up[3 * m - 3] = 1
        vec_up[((3 * m) % (3 * L)) + 0] = -1  # mod(3*m,3*L)+1 -> 0-based

        vec_dn = np.zeros(3 * L, dtype=int)
        vec_dn[3 * m - 2] = 1
        vec_dn[((3 * m) % (3 * L)) + 1] = -1  # mod(3*m,3*L)+2 -> 0-based

        # build COO entries
        tu_rows, tu_cols, tu_vals = [], [], []
        td_rows, td_cols, td_vals = [], [], []
        bd_rows, bd_cols, bd_vals = [], [], []
        b_rows, b_cols, b_vals = [], [], []
        nup_diag = np.zeros(dim, dtype=np.float64)
        ndn_diag = np.zeros(dim, dtype=np.float64)
        nph_diag = np.zeros(dim, dtype=np.float64)

        # loop over basis labels (MATLAB 1..dim)
        for cnt in range(1, dim + 1):
            psi = label2fock(cnt, L, filling, Nmax)

            # total spin-up occupancy for boundary sign (as in MATLAB)
            occupat = int(psi[0::3].sum())

            nup_diag[cnt - 1] = psi[3 * m - 3]
            ndn_diag[cnt - 1] = psi[3 * m - 2]
            nph_diag[cnt - 1] = psi[3 * m - 1]

            # phonon creation
            phi = psi + ladder_vec
            if 0 <= phi[3 * m - 1] <= Nmax:
                tnc = fock2label(phi, L, filling, Nmax)
                bd_rows.append(cnt - 1)
                bd_cols.append(tnc - 1)
                bd_vals.append(np.sqrt(psi[3 * m - 1] + 1.0))

            # phonon annihilation
            phi = psi - ladder_vec
            if 0 <= phi[3 * m - 1] <= Nmax:
                tnc = fock2label(phi, L, filling, Nmax)
                b_rows.append(cnt - 1)
                b_cols.append(tnc - 1)
                b_vals.append(np.sqrt(float(psi[3 * m - 1])))

            # spin-up hopping
            phi = psi + vec_up
            ok = (phi[0::3].max() <= 1 and phi[1::3].max() <= 1 and phi[2::3].max() <= Nmax and phi.min() >= 0)
            if ok:
                tnc = fock2label(phi, L, filling, Nmax)
                # add both directions like MATLAB
                tu_rows.extend([cnt - 1, tnc - 1])
                tu_cols.extend([tnc - 1, cnt - 1])

                # boundary sign
                if m == L:
                    a = -np.sign(m - L + 0.5) * ((-1) ** occupat)
                else:
                    a = -np.sign(m - L + 0.5)
                tu_vals.extend([a, a])

            # spin-down hopping
            phi = psi + vec_dn
            ok = (phi[1::3].max() <= 1 and phi[0::3].max() <= 1 and phi[2::3].max() <= Nmax and phi.min() >= 0)
            if ok:
                tnc = fock2label(phi, L, filling, Nmax)
                td_rows.extend([cnt - 1, tnc - 1])
                td_cols.extend([tnc - 1, cnt - 1])

                if m == L:
                    a = -np.sign(m - L + 0.5) * ((-1) ** occupat)
                else:
                    a = -np.sign(m - L + 0.5)
                td_vals.extend([a, a])

        tunn_up.append(sp.csr_matrix((np.array(tu_vals, dtype=np.complex128),
                                      (np.array(tu_rows), np.array(tu_cols))),
                                     shape=(dim, dim)))
        tunn_dn.append(sp.csr_matrix((np.array(td_vals, dtype=np.complex128),
                                      (np.array(td_rows), np.array(td_cols))),
                                     shape=(dim, dim)))
        bdag.append(sp.csr_matrix((np.array(bd_vals, dtype=np.complex128),
                                   (np.array(bd_rows), np.array(bd_cols))),
                                  shape=(dim, dim)))
        b.append(sp.csr_matrix((np.array(b_vals, dtype=np.complex128),
                                (np.array(b_rows), np.array(b_cols))),
                               shape=(dim, dim)))

        nup.append(sp.diags(nup_diag, 0, format="csr", dtype=np.complex128))
        ndn.append(sp.diags(ndn_diag, 0, format="csr", dtype=np.complex128))
        nph.append(sp.diags(nph_diag, 0, format="csr", dtype=np.complex128))

    return LocalOperators(
        tunneling_up=tunn_up,
        tunneling_down=tunn_dn,
        phonon_creator=bdag,
        phonon_annihilator=b,
        n_up=nup,
        n_down=ndn,
        n_ph=nph,
    )
