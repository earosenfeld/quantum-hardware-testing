"""Validation tests for single-shot readout assignment fidelity."""

import numpy as np
import pytest

from src.qubit.readout import simulate_readout_iq, assignment_fidelity


def test_high_snr_fidelity_approaches_one():
    """Well-separated IQ blobs give assignment fidelity near 1."""
    iq0, iq1 = simulate_readout_iq(separation=8.0, sigma=1.0, n_shots=20000, seed=0)
    res = assignment_fidelity(iq0, iq1)
    assert res.fidelity > 0.99
    assert res.f0 > 0.99 and res.f1 > 0.99


def test_overlapping_blobs_fidelity_approaches_half():
    """Fully overlapping blobs carry no information: F -> 0.5."""
    iq0, iq1 = simulate_readout_iq(separation=0.0, sigma=1.0, n_shots=20000, seed=1)
    res = assignment_fidelity(iq0, iq1)
    assert abs(res.fidelity - 0.5) < 0.03


def test_confusion_matrix_rows_normalised():
    iq0, iq1 = simulate_readout_iq(separation=4.0, sigma=1.0, n_shots=10000, seed=2)
    res = assignment_fidelity(iq0, iq1)
    assert res.confusion.shape == (2, 2)
    np.testing.assert_allclose(res.confusion.sum(axis=1), [1.0, 1.0], atol=1e-9)


def test_fidelity_formula_matches_confusion():
    iq0, iq1 = simulate_readout_iq(separation=3.0, sigma=1.0, n_shots=15000, seed=3)
    res = assignment_fidelity(iq0, iq1)
    expected = 1.0 - 0.5 * (res.p1_given_0 + res.p0_given_1)
    assert res.fidelity == pytest.approx(expected)


def test_fidelity_monotonic_in_snr():
    """Increasing separation (SNR) monotonically improves fidelity."""
    fids = []
    for sep in (1.0, 2.0, 4.0, 8.0):
        iq0, iq1 = simulate_readout_iq(separation=sep, sigma=1.0, n_shots=20000, seed=4)
        fids.append(assignment_fidelity(iq0, iq1).fidelity)
    assert all(b >= a - 1e-3 for a, b in zip(fids, fids[1:]))


def test_discriminator_works_at_arbitrary_angle():
    """The discriminator must not assume blobs lie along the I axis."""
    iq0, iq1 = simulate_readout_iq(
        separation=6.0, sigma=1.0, n_shots=20000, angle=0.9, seed=5
    )
    res = assignment_fidelity(iq0, iq1)
    assert res.fidelity > 0.99
