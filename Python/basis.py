
"""
Basis utilities for the electron+Einstein-phonon ED code.

The original MATLAB code uses a fixed-particle-number fermionic sector with
2*L spin-orbitals (up1..upL, down1..downL) and a local phonon cutoff Nmax per site.

The full basis is ordered as:
  |fermion_configuration_index> ⊗ |phonon_configuration_index>

with:
  fermion_configuration_index in [1, C(2L, filling)]
  phonon_configuration_index  in [1, (Nmax+1)^L]

The global label is (MATLAB 1-based):
  label = (fermion_index - 1) * (Nmax+1)^L + phonon_index
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
import numpy as np


def _nchoosek(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return comb(n, k)


def fock2label(psi: np.ndarray, L: int, filling: int, Nmax: int) -> int:
    """
    Convert a Fock-state occupation vector `psi` to a 1-based Hilbert-space label.

    psi is length 3*L and organized as:
      [up1, down1, phonon1, up2, down2, phonon2, ..., upL, downL, phononL]

    This matches the MATLAB implementation in fock2label.m.
    """
    psi = np.asarray(psi, dtype=int).ravel()
    if psi.size != 3 * L:
        raise ValueError(f"psi must have length 3*L={3*L}, got {psi.size}")

    psi_up = psi[0::3]
    psi_dn = psi[1::3]
    psi_ph = psi[2::3]

    psi_matter = np.concatenate([psi_up, psi_dn])  # length 2L

    if int(psi_matter.sum()) != int(filling):
        # MATLAB returns -1 and prints; we raise.
        raise ValueError("Input state is not in the fixed-N (filling) sector.")

    # Fermion (matter) combinatorial labeling (MATLAB-style).
    n = int(filling)  # remaining fermions to place
    m = 1            # 1-based label within matter block

    for s in range(1, 2 * L + 1):  # MATLAB s=1..2L
        if n == 0:
            break
        if psi_matter[s - 1] == 0:
            m += _nchoosek(2 * L - s, n - 1)
        else:
            n -= 1

    # Phonons: base-(Nmax+1) digits, most significant = site 1.
    base = Nmax + 1
    if np.any(psi_ph < 0) or np.any(psi_ph > Nmax):
        raise ValueError("Phonon occupations out of range [0, Nmax].")
    ph_index0 = 0
    for s in range(L):
        ph_index0 = ph_index0 * base + int(psi_ph[s])
    phonon_index = ph_index0 + 1  # 1-based

    label = (m - 1) * (base ** L) + phonon_index
    return int(label)


def label2fock(cnt: int, L: int, filling: int, Nmax: int) -> np.ndarray:
    """
    Inverse of fock2label: convert 1-based label `cnt` to a Fock vector psi (length 3*L).
    Matches the MATLAB implementation in label2fock.m.
    """
    cnt = int(cnt)
    if cnt < 1:
        raise ValueError("cnt must be >= 1 (MATLAB 1-based).")

    base = Nmax + 1
    ph_block = base ** L

    cnt_matter = int(np.ceil(cnt / ph_block))
    cnt_phonons = (cnt - 1) % ph_block  # 0-based within phonon block

    # Decode phonons: L digits in base, MSB=site1
    psi_ph = np.zeros(L, dtype=int)
    x = cnt_phonons
    for s in range(L - 1, -1, -1):
        psi_ph[s] = x % base
        x //= base

    # Decode matter index using the inverse of MATLAB's combinatorial labeling.
    m = cnt_matter
    n = int(filling)
    psi_matter = np.zeros(2 * L, dtype=int)

    for s in range(1, 2 * L + 1):
        if n == 0:
            break

        btmp = _nchoosek(2 * L - s, n - 1)
        if m > btmp:
            # orbital empty
            m -= btmp
            psi_matter[s - 1] = 0
        else:
            # orbital occupied
            psi_matter[s - 1] = 1
            n -= 1

    # Pack back into [up, down, ph] per site
    psi = np.zeros(3 * L, dtype=int)
    for i in range(L):
        psi[3 * i]     = psi_matter[i]       # up_i
        psi[3 * i + 1] = psi_matter[i + L]   # down_i
        psi[3 * i + 2] = psi_ph[i]           # phonon_i
    return psi
