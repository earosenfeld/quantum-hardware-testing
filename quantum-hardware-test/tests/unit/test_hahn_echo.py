"""Validation tests for the Hahn-echo T2 experiment."""

import numpy as np
import pytest

from src.qubit.hahn_echo import simulate_hahn_echo, fit_hahn_echo


def test_echo_extends_coherence_beyond_ramsey():
    """For the same quasi-static noise, T2_echo > T2* (refocusing works)."""
    sigma_f = 80e3  # 80 kHz quasi-static frequency noise
    t2_intrinsic = 100e-6

    # Use a common delay window so both decays are sampled identically.
    delays = np.linspace(0.0, 60e-6, 60)
    d_r, p_r, s_r = simulate_hahn_echo(
        sigma_f, t2_intrinsic, echo=False, delays=delays, n_shots=16384, seed=0
    )
    d_e, p_e, s_e = simulate_hahn_echo(
        sigma_f, t2_intrinsic, echo=True, delays=delays, n_shots=16384, seed=1
    )
    t2_star = fit_hahn_echo(d_r, p_r, s_r).T1
    t2_echo = fit_hahn_echo(d_e, p_e, s_e).T1

    assert t2_echo > t2_star, f"echo {t2_echo:.2e} !> ramsey {t2_star:.2e}"


def test_echo_recovers_intrinsic_t2():
    """With a perfect refocusing pulse the echo decays at the intrinsic time."""
    sigma_f = 120e3
    t2_intrinsic = 90e-6
    delays = np.linspace(0.0, 3 * t2_intrinsic, 60)
    d, p, s = simulate_hahn_echo(
        sigma_f, t2_intrinsic, echo=True, delays=delays, n_shots=32000, seed=2
    )
    res = fit_hahn_echo(d, p, s)
    assert abs(res.T1 - t2_intrinsic) / t2_intrinsic < 0.05


@pytest.mark.parametrize("sigma_f", [50e3, 150e3, 300e3])
def test_echo_gain_grows_with_noise(sigma_f):
    """Stronger low-frequency noise -> larger echo/Ramsey coherence ratio."""
    t2_intrinsic = 100e-6
    delays = np.linspace(0.0, 40e-6, 60)
    _, p_r, s_r = simulate_hahn_echo(
        sigma_f, t2_intrinsic, echo=False, delays=delays, n_shots=16384, seed=3
    )
    _, p_e, s_e = simulate_hahn_echo(
        sigma_f, t2_intrinsic, echo=True, delays=delays, n_shots=16384, seed=4
    )
    t2_star = fit_hahn_echo(delays, p_r, s_r).T1
    t2_echo = fit_hahn_echo(delays, p_e, s_e).T1
    assert t2_echo > t2_star
