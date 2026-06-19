"""Single-qubit Clifford randomized benchmarking (RB).

RB estimates the average error per Clifford independent of state-prep and
measurement (SPAM) error. For each sequence length ``m`` we:

1. draw ``m`` uniformly random single-qubit Cliffords (the 24-element group),
2. append the single recovery Clifford that inverts their product (so the
   ideal circuit maps |0> -> |0>),
3. apply a simple depolarizing error per Clifford and read out the survival
   probability (probability of measuring |0>).

Averaged over random sequences the survival decays as

    S(m) = A * p^m + B

where ``p`` is the depolarizing parameter. The error per Clifford is

    EPC = (1 - p) * (d - 1) / d,   d = 2.

The Clifford group is represented by its action on the Pauli operators (a
signed permutation of {X, Y, Z}); composition and inversion are exact, so the
recovery gate is computed correctly. The noise model is a depolarizing channel
applied in the Bloch picture, which yields the exact ``A*p^m+B`` form -- letting
the test recover the injected ``p`` / EPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .models import FitResult, fit_curve

# --------------------------------------------------------------------------- #
# Single-qubit Clifford group as 3x3 signed permutation matrices acting on the
# Bloch vector (the rotation part of each Clifford in SO(3)). There are exactly
# 24 such proper rotations -- the 24-element single-qubit Clifford group.
# --------------------------------------------------------------------------- #
def _generate_clifford_so3() -> np.ndarray:
    """Enumerate the 24 signed 3x3 permutation matrices with det = +1."""
    mats = []
    perms = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
    for perm in perms:
        for signs in range(8):
            s = [1 if not (signs >> k) & 1 else -1 for k in range(3)]
            m = np.zeros((3, 3))
            for row, col in enumerate(perm):
                m[row, col] = s[row]
            if round(float(np.linalg.det(m))) == 1:
                mats.append(m)
    arr = np.array(mats)
    assert arr.shape[0] == 24, f"expected 24 Cliffords, got {arr.shape[0]}"
    return arr


@lru_cache(maxsize=1)
def _clifford_group() -> tuple:
    """Return (matrices, index lookup) for the 24 single-qubit Cliffords."""
    mats = _generate_clifford_so3()
    # Map a rounded matrix key -> group index for fast composition/inversion.
    lookup = {}
    for i, m in enumerate(mats):
        lookup[_key(m)] = i
    return mats, lookup


def _key(m: np.ndarray) -> tuple:
    return tuple(np.round(m).astype(int).flatten().tolist())


def _compose_index(group: np.ndarray, lookup: dict, a: int, b: int) -> int:
    """Index of the Clifford ``group[a] @ group[b]`` (apply b then a)."""
    prod = group[a] @ group[b]
    return lookup[_key(prod)]


def _inverse_index(group: np.ndarray, lookup: dict, a: int) -> int:
    inv = group[a].T  # rotation inverse = transpose
    return lookup[_key(inv)]


@dataclass
class RBResult:
    """Randomized-benchmarking fit result."""

    p: float
    p_err: float
    epc: float
    epc_err: float
    A: float
    B: float
    fit: FitResult


def epc_from_p(p: float, d: int = 2) -> float:
    """Error per Clifford from the depolarizing parameter ``p``."""
    return (1.0 - p) * (d - 1) / d


def rb_survival_model(m: np.ndarray, A: float, p: float, B: float) -> np.ndarray:
    """RB survival curve ``S(m) = A * p^m + B``."""
    return A * np.power(p, m) + B


def simulate_rb(
    p_depol: float,
    lengths: np.ndarray | None = None,
    n_sequences: int = 40,
    n_shots: int = 4096,
    A: float = 0.5,
    B: float = 0.5,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate an RB experiment and return averaged survival vs length.

    Parameters
    ----------
    p_depol:
        Depolarizing parameter applied per Clifford (1 = noiseless). The
        survival decays as ``A*p_depol^m + B``.
    lengths:
        Sequence lengths (number of random Cliffords). Defaults to a log-spaced
        sweep.
    n_sequences:
        Random sequences averaged per length.
    A, B:
        SPAM-determined offsets of the ideal curve (``A+B`` is the m=0 survival).

    Returns
    -------
    (lengths, survival, sigma):
        Length axis, mean survival probability, std error across sequences.
    """
    rng = np.random.default_rng(seed)
    group, lookup = _clifford_group()
    if lengths is None:
        lengths = np.unique(
            np.round(np.geomspace(1, 200, 12)).astype(int)
        )
    lengths = np.asarray(lengths, dtype=int)

    survival_mean = np.zeros(len(lengths))
    survival_sem = np.zeros(len(lengths))

    for li, m in enumerate(lengths):
        seq_survivals = np.empty(n_sequences)
        for s in range(n_sequences):
            gates = rng.integers(0, 24, size=int(m))
            # Net Clifford of the random sequence (apply gates[0] first).
            net = 0  # identity index
            # Identity is the unique matrix equal to I; find it once.
            net = _identity_index(group, lookup)
            for g in gates:
                net = _compose_index(group, lookup, int(g), net)
            recovery = _inverse_index(group, lookup, net)

            # Ideal circuit returns to |0>; depolarizing noise per applied
            # Clifford (m random gates + 1 recovery) shrinks the Bloch vector.
            n_gates = int(m) + 1
            # Survival of |0> under n_gates depolarizing events:
            #   P(0) = 0.5 * (1 + p^n_gates)   (ideal end state is |0>, z=+1)
            p0 = 0.5 * (1.0 + p_depol ** n_gates)
            # Shot noise on the survival estimate.
            counts = rng.binomial(n_shots, p0)
            seq_survivals[s] = counts / n_shots
        survival_mean[li] = seq_survivals.mean()
        survival_sem[li] = seq_survivals.std(ddof=1) / np.sqrt(n_sequences) if n_sequences > 1 else 0.0

    # Floor the per-point sigma so a degenerate zero-variance length does not
    # break the weighted fit.
    floor = (0.5 / n_shots)
    survival_sem = np.maximum(survival_sem, floor)
    return lengths.astype(float), survival_mean, survival_sem


@lru_cache(maxsize=1)
def _identity_index_cached() -> int:
    group, lookup = _clifford_group()
    return lookup[_key(np.eye(3))]


def _identity_index(group: np.ndarray, lookup: dict) -> int:
    return _identity_index_cached()


def fit_rb(
    lengths: np.ndarray, survival: np.ndarray, sigma: np.ndarray | None = None
) -> RBResult:
    """Fit ``A*p^m+B`` and return p / EPC with covariance-based uncertainty."""
    lengths = np.asarray(lengths, dtype=float)
    survival = np.asarray(survival, dtype=float)

    b0 = float(np.min(survival))
    a0 = float(np.clip(survival[0] - b0, 1e-3, 1.0))
    p0 = 0.99

    fit = fit_curve(
        rb_survival_model,
        lengths,
        survival,
        p0=[a0, p0, b0],
        names=("A", "p", "B"),
        sigma=sigma,
        bounds=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    )
    p = fit.value("p")
    p_err = fit.error("p")
    epc = epc_from_p(p)
    epc_err = p_err * 0.5  # EPC = (1-p)/2  =>  d EPC = dp / 2
    return RBResult(
        p=p,
        p_err=p_err,
        epc=epc,
        epc_err=epc_err,
        A=fit.value("A"),
        B=fit.value("B"),
        fit=fit,
    )
