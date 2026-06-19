"""Validation tests for single-qubit randomized benchmarking."""

import numpy as np
import pytest

from src.qubit.randomized_benchmarking import (
    simulate_rb,
    fit_rb,
    epc_from_p,
    rb_survival_model,
    _clifford_group,
    _compose_index,
    _inverse_index,
    _identity_index,
)


def test_clifford_group_has_24_elements():
    group, lookup = _clifford_group()
    assert len(group) == 24
    assert len(lookup) == 24


def test_clifford_group_closed_under_composition():
    group, lookup = _clifford_group()
    for a in range(24):
        for b in range(24):
            # Composition index must exist (group closure).
            idx = _compose_index(group, lookup, a, b)
            assert 0 <= idx < 24


def test_clifford_inverse_is_well_defined():
    group, lookup = _clifford_group()
    ident = _identity_index(group, lookup)
    for a in range(24):
        inv = _inverse_index(group, lookup, a)
        # a composed with its inverse is the identity.
        assert _compose_index(group, lookup, a, inv) == ident


def test_rb_recovers_depolarizing_parameter():
    p_true = 0.99
    lengths, survival, sigma = simulate_rb(
        p_true, n_sequences=30, n_shots=4096, seed=0
    )
    res = fit_rb(lengths, survival, sigma)
    assert abs(res.p - p_true) < 0.005
    assert res.epc == pytest.approx(epc_from_p(p_true), abs=2.5e-3)


def test_rb_epc_matches_formula():
    assert epc_from_p(0.99) == pytest.approx(0.005)
    assert epc_from_p(1.0) == pytest.approx(0.0)


def test_rb_survival_decreasing_with_length():
    p_true = 0.95
    lengths, survival, _ = simulate_rb(p_true, n_sequences=20, n_shots=8192, seed=1)
    # Survival trends down with sequence length (monotone in expectation).
    assert survival[0] > survival[-1]


def test_rb_model_shape():
    m = np.array([0, 1, 10, 100], dtype=float)
    y = rb_survival_model(m, A=0.5, p=0.99, B=0.5)
    assert y[0] == pytest.approx(1.0)
    assert np.all(np.diff(y) <= 0)
