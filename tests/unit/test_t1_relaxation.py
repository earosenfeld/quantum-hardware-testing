"""Validation tests for the T1 relaxation experiment."""

import numpy as np
import pytest

from qht.qubit.relaxation import simulate_t1, fit_t1
from qht.qubit.models import exp_decay


def test_t1_recovered_within_error_bar():
    """Injected T1 is recovered within ~3 sigma of the covariance error bar."""
    t1_true = 50e-6
    delays, p_hat, sigma = simulate_t1(t1_true, n_shots=4096, seed=0)
    res = fit_t1(delays, p_hat, sigma)

    assert res.T1_err > 0
    n_sigma = abs(res.T1 - t1_true) / res.T1_err
    assert n_sigma < 3.0, f"T1 off by {n_sigma:.2f} sigma"


def test_t1_high_shots_few_percent_accuracy():
    """At high shot count the fit lands within a few percent of truth."""
    t1_true = 80e-6
    delays, p_hat, sigma = simulate_t1(t1_true, n_shots=50000, seed=1)
    res = fit_t1(delays, p_hat, sigma)

    rel_err = abs(res.T1 - t1_true) / t1_true
    assert rel_err < 0.03, f"relative error {rel_err:.3%} too large"


def test_t1_uncertainty_shrinks_with_more_shots():
    """More shots -> smaller covariance error bar (sqrt(N) scaling, roughly)."""
    t1_true = 40e-6
    _, p_lo, s_lo = simulate_t1(t1_true, n_shots=1024, seed=2)
    _, p_hi, s_hi = simulate_t1(t1_true, n_shots=16384, seed=2)
    delays, _, _ = simulate_t1(t1_true, n_shots=1024, seed=2)
    err_lo = fit_t1(delays, p_lo, s_lo).T1_err
    err_hi = fit_t1(delays, p_hi, s_hi).T1_err
    assert err_hi < err_lo


def test_t1_model_is_monotonic_decay():
    t = np.linspace(0, 1e-4, 50)
    y = exp_decay(t, A=1.0, T1=30e-6, C=0.0)
    assert np.all(np.diff(y) <= 0)
    assert y[0] == pytest.approx(1.0)


@pytest.mark.parametrize("t1_true", [10e-6, 35e-6, 120e-6])
def test_t1_recovered_across_scales(t1_true):
    delays, p_hat, sigma = simulate_t1(t1_true, n_shots=20000, seed=7)
    res = fit_t1(delays, p_hat, sigma)
    assert abs(res.T1 - t1_true) / t1_true < 0.05
