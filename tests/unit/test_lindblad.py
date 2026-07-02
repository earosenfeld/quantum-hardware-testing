"""Tests for the optional QuTiP Lindblad master-equation engine.

These are SKIPPED (not failed) when QuTiP is not installed, so the suite stays
green on a numpy/scipy-only environment.
"""

import numpy as np
import pytest

from qht.qubit.lindblad import have_qutip

pytestmark = pytest.mark.skipif(not have_qutip(), reason="qutip not installed")


def test_t1_emerges_from_master_equation():
    """Excited population from |1> decays at the injected T1 (no Tphi)."""
    from qht.qubit.lindblad import simulate_t1_lindblad
    from qht.qubit.relaxation import fit_t1

    t1 = 60e-6
    tlist, p1 = simulate_t1_lindblad(t1, tphi=np.inf)
    res = fit_t1(tlist, p1)
    assert abs(res.T1 - t1) / t1 < 0.02


def test_t2_emerges_and_obeys_relation():
    """Coherence from |+> decays at 1/T2 = 1/(2 T1) + 1/Tphi."""
    from qht.qubit.lindblad import simulate_t2_lindblad
    from qht.qubit.hahn_echo import fit_hahn_echo

    t1, tphi = 60e-6, 40e-6
    tlist, coh, t2_expected = simulate_t2_lindblad(t1, tphi)
    res = fit_hahn_echo(tlist, coh)
    assert abs(res.T1 - t2_expected) / t2_expected < 0.03

    analytic = 1.0 / (1.0 / (2.0 * t1) + 1.0 / tphi)
    assert t2_expected == pytest.approx(analytic, rel=1e-9)


def test_collapse_operators_have_correct_rates():
    """c1 = sqrt(Gamma1) sigma^-, cphi = sqrt(Gamma_phi/2) sigma_z."""
    import qutip as qt
    from qht.qubit.lindblad import collapse_operators

    t1, tphi = 50e-6, 30e-6
    c_ops = collapse_operators(t1, tphi)
    assert len(c_ops) == 2

    gamma1 = 1.0 / t1
    gamma_phi = 1.0 / tphi
    # ||c1||^2 coefficient should equal Gamma1 (sigma^- has unit norm entries).
    c1_coef = (c_ops[0].dag() * c_ops[0]).tr() / (qt.sigmam().dag() * qt.sigmam()).tr()
    cphi_coef = (c_ops[1].dag() * c_ops[1]).tr() / (qt.sigmaz().dag() * qt.sigmaz()).tr()
    assert float(np.real(c1_coef)) == pytest.approx(gamma1, rel=1e-9)
    assert float(np.real(cphi_coef)) == pytest.approx(gamma_phi / 2.0, rel=1e-9)
