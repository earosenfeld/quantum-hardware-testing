"""Tests for the shared fit machinery and covariance-based uncertainties."""

import numpy as np
import pytest

from src.qubit.models import (
    FitResult,
    exp_decay,
    ramsey_decay,
    rabi_cosine,
    fit_curve,
    sample_shots,
)


def test_perr_is_sqrt_diag_pcov():
    """The reported uncertainties are exactly sqrt(diag(pcov))."""
    t = np.linspace(0, 1e-4, 60)
    y = exp_decay(t, 1.0, 40e-6, 0.0) + np.random.default_rng(0).normal(0, 0.01, t.size)
    res = fit_curve(exp_decay, t, y, p0=[1.0, 30e-6, 0.0], names=("A", "T1", "C"))
    np.testing.assert_allclose(res.perr, np.sqrt(np.diag(res.pcov)))


def test_fitresult_labelled_lookup():
    t = np.linspace(0, 1e-4, 60)
    y = exp_decay(t, 0.9, 50e-6, 0.05)
    res = fit_curve(exp_decay, t, y, p0=[1.0, 40e-6, 0.0], names=("A", "T1", "C"))
    assert isinstance(res, FitResult)
    assert res.value("T1") == pytest.approx(50e-6, rel=1e-3)
    d = res.as_dict()
    assert "stderr" in d["T1"] and "value" in d["T1"]


def test_weighted_fit_absolute_sigma_scales_errors():
    """Larger per-point sigma -> larger covariance error bars (absolute_sigma)."""
    t = np.linspace(0, 1e-4, 60)
    y = exp_decay(t, 1.0, 40e-6, 0.0)
    s_small = np.full_like(t, 0.005)
    s_large = np.full_like(t, 0.05)
    e_small = fit_curve(
        exp_decay, t, y, p0=[1, 40e-6, 0], names=("A", "T1", "C"), sigma=s_small
    ).error("T1")
    e_large = fit_curve(
        exp_decay, t, y, p0=[1, 40e-6, 0], names=("A", "T1", "C"), sigma=s_large
    ).error("T1")
    assert e_large > e_small


def test_sample_shots_binomial_statistics():
    """sample_shots returns proportions near truth with binomial std error."""
    rng = np.random.default_rng(42)
    p = np.array([0.2, 0.5, 0.8])
    p_hat, sigma = sample_shots(p, n_shots=100000, rng=rng)
    np.testing.assert_allclose(p_hat, p, atol=0.01)
    expected_sigma = np.sqrt(p * (1 - p) / 100000)
    np.testing.assert_allclose(sigma, expected_sigma, rtol=0.1)


def test_sample_shots_handles_extremes():
    """P=0 and P=1 still produce finite (floored) error bars."""
    rng = np.random.default_rng(0)
    p_hat, sigma = sample_shots(np.array([0.0, 1.0]), n_shots=1000, rng=rng)
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0)


def test_model_functions_basic_values():
    assert exp_decay(np.array([0.0]), 1.0, 1.0, 0.0)[0] == pytest.approx(1.0)
    # Ramsey at t=0, phi=0 -> A + C
    assert ramsey_decay(np.array([0.0]), 0.5, 1.0, 1.0, 0.0, 0.5)[0] == pytest.approx(1.0)
    # Rabi at the pi pulse tau = 1/(2 f_rabi) -> A + C
    f = 1e6
    assert rabi_cosine(np.array([1 / (2 * f)]), 1.0, f, 0.0)[0] == pytest.approx(1.0)
