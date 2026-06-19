"""Optional QuTiP Lindblad master-equation engine.

This module lets T1 and T2 *emerge* from open-system dynamics instead of being
assumed by a closed-form formula. It is entirely optional: it imports QuTiP
lazily inside the functions so that importing :mod:`src.qubit` (and running the
whole test suite) works with only numpy/scipy installed. Call :func:`have_qutip`
to check availability before using the engine; tests skip these paths when it
is absent.

Physics (single qubit, two collapse operators):

- Energy relaxation:  c1 = sqrt(Gamma1) * sigma^-,  Gamma1 = 1 / T1
- Pure dephasing:      cphi = sqrt(Gamma_phi / 2) * sigma_z

with the standard relation

    1 / T2 = 1 / (2 * T1) + 1 / T_phi.

Evolving the appropriate initial state under ``qutip.mesolve`` reproduces:

- T1: from |1>, <sigma_z> (excited population) decays as exp(-t/T1);
- T2: from |+>, the coherence <sigma_x> decays as exp(-t/T2).

Fitting those simulated curves recovers the input T1/T2, demonstrating the
closed-form models used elsewhere are the master-equation result.
"""

from __future__ import annotations

import numpy as np


def have_qutip() -> bool:
    """True if QuTiP is importable in this environment."""
    try:
        import qutip  # noqa: F401

        return True
    except Exception:
        return False


def _rates(t1: float, tphi: float) -> tuple[float, float, float]:
    gamma1 = 1.0 / t1
    gamma_phi = 1.0 / tphi if np.isfinite(tphi) and tphi > 0 else 0.0
    t2 = 1.0 / (1.0 / (2.0 * t1) + (1.0 / tphi if gamma_phi > 0 else 0.0))
    return gamma1, gamma_phi, t2


def collapse_operators(t1: float, tphi: float):
    """Return the [c1, cphi] collapse operators for the given T1, Tphi.

    Raises ``ImportError`` if QuTiP is unavailable.
    """
    import qutip as qt

    gamma1, gamma_phi, _ = _rates(t1, tphi)
    c_ops = [np.sqrt(gamma1) * qt.sigmam()]
    if gamma_phi > 0:
        c_ops.append(np.sqrt(gamma_phi / 2.0) * qt.sigmaz())
    return c_ops


def simulate_t1_lindblad(
    t1: float, tphi: float = np.inf, tlist: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Master-equation T1 decay: excited population P1(t) from |1>.

    Returns ``(tlist, P1)`` where ``P1 = (1 + <sigma_z>)/2``.
    """
    import qutip as qt

    if tlist is None:
        tlist = np.linspace(0.0, 4.0 * t1, 60)
    H = qt.qzero(2)  # rotating frame, on resonance: no coherent evolution
    psi0 = qt.basis(2, 0)  # |0> is the excited state in qutip's sigmaz convention
    # qutip sigmam() = |0><1|; basis(2,1) is the lower state. Use the excited
    # state (eigenvalue +1 of sigmaz) so population decays. basis(2,0) has
    # sigmaz expectation +1.
    c_ops = collapse_operators(t1, tphi)
    result = qt.mesolve(H, psi0, tlist, c_ops, e_ops=[qt.sigmaz()])
    sz = np.asarray(result.expect[0])
    p1 = 0.5 * (1.0 + sz)
    return np.asarray(tlist), p1


def simulate_t2_lindblad(
    t1: float, tphi: float, tlist: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Master-equation T2 decay: coherence <sigma_x>(t) from |+>.

    Returns ``(tlist, coherence, t2_expected)`` where ``coherence`` decays as
    ``exp(-t/T2)`` and ``t2_expected`` is the analytic 1/T2 = 1/(2T1)+1/Tphi.
    """
    import qutip as qt

    _, _, t2 = _rates(t1, tphi)
    if tlist is None:
        tlist = np.linspace(0.0, 4.0 * t2, 60)
    H = qt.qzero(2)
    psi0 = (qt.basis(2, 0) + qt.basis(2, 1)).unit()  # |+>
    c_ops = collapse_operators(t1, tphi)
    result = qt.mesolve(H, psi0, tlist, c_ops, e_ops=[qt.sigmax()])
    coh = np.asarray(result.expect[0])
    return np.asarray(tlist), coh, t2
