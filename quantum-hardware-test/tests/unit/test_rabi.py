"""Validation tests for the Rabi calibration experiment."""

import numpy as np
import pytest

from src.qubit.rabi import simulate_rabi, fit_rabi


def test_rabi_recovers_frequency_and_pi_pulse():
    """Rabi frequency and the derived pi-pulse duration are recovered."""
    f_rabi_true = 10e6  # 10 MHz Rabi rate
    t_pi_true = 1.0 / (2.0 * f_rabi_true)
    durations, p_hat, sigma = simulate_rabi(f_rabi_true, n_shots=4096, seed=0)
    res = fit_rabi(durations, p_hat, sigma)

    assert abs(res.f_rabi - f_rabi_true) / f_rabi_true < 0.02
    assert abs(res.t_pi - t_pi_true) / t_pi_true < 0.02


def test_rabi_pi_pulse_uncertainty_propagated():
    f_rabi_true = 5e6
    durations, p_hat, sigma = simulate_rabi(f_rabi_true, n_shots=8192, seed=2)
    res = fit_rabi(durations, p_hat, sigma)
    # t_pi error is the fractional frequency error scaled onto t_pi.
    assert res.t_pi_err > 0
    expected = res.t_pi * (res.f_rabi_err / res.f_rabi)
    assert res.t_pi_err == pytest.approx(expected, rel=1e-9)


def test_rabi_pi_pulse_within_error_bar():
    f_rabi_true = 8e6
    t_pi_true = 1.0 / (2.0 * f_rabi_true)
    durations, p_hat, sigma = simulate_rabi(f_rabi_true, n_shots=16384, seed=4)
    res = fit_rabi(durations, p_hat, sigma)
    assert abs(res.t_pi - t_pi_true) / res.t_pi_err < 3.5


@pytest.mark.parametrize("f_rabi_true", [2e6, 12e6, 25e6])
def test_rabi_frequency_scan(f_rabi_true):
    durations, p_hat, sigma = simulate_rabi(f_rabi_true, n_shots=20000, seed=9)
    res = fit_rabi(durations, p_hat, sigma)
    assert abs(res.f_rabi - f_rabi_true) / f_rabi_true < 0.03
