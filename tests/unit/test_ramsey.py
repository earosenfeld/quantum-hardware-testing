"""Validation tests for the Ramsey T2* experiment."""

import numpy as np
import pytest

from qht.qubit.ramsey import simulate_ramsey, fit_ramsey


def test_ramsey_recovers_t2_and_detuning():
    """Both T2* and the detuning are recovered from the fringe fit."""
    t2_true = 30e-6
    df_true = 0.5e6  # 500 kHz detuning
    delays, p_hat, sigma = simulate_ramsey(t2_true, df_true, n_shots=4096, seed=0)
    res = fit_ramsey(delays, p_hat, sigma)

    assert abs(res.T2 - t2_true) / t2_true < 0.10
    assert abs(res.delta_f - df_true) / df_true < 0.02


def test_ramsey_t2_within_error_bar():
    t2_true = 25e-6
    df_true = 0.8e6
    delays, p_hat, sigma = simulate_ramsey(t2_true, df_true, n_shots=8192, seed=3)
    res = fit_ramsey(delays, p_hat, sigma)
    assert res.T2_err > 0 and res.delta_f_err > 0
    assert abs(res.T2 - t2_true) / res.T2_err < 3.5


@pytest.mark.parametrize("df_true", [0.2e6, 1.0e6, 2.0e6])
def test_ramsey_detuning_scan(df_true):
    t2_true = 40e-6
    delays, p_hat, sigma = simulate_ramsey(t2_true, df_true, n_shots=8192, seed=11)
    res = fit_ramsey(delays, p_hat, sigma)
    assert abs(res.delta_f - df_true) / df_true < 0.05


def test_ramsey_detuning_high_shots_accuracy():
    t2_true = 35e-6
    df_true = 0.6e6
    delays, p_hat, sigma = simulate_ramsey(t2_true, df_true, n_shots=50000, seed=5)
    res = fit_ramsey(delays, p_hat, sigma)
    assert abs(res.delta_f - df_true) / df_true < 0.01
