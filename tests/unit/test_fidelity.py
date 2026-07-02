"""Validation tests for gate fidelity and the coherence-limited error budget."""

import pytest

from src.qubit.fidelity import (
    average_gate_fidelity_from_rb,
    error_per_clifford_from_rb,
    coherence_limit_error,
    error_budget,
)


# --- average gate fidelity from RB ----------------------------------------- #
def test_gate_fidelity_matches_closed_form_single_qubit():
    # p = 0.99, d = 2  ->  r = (0.01)(1/2) = 0.005,  F = 0.995
    assert error_per_clifford_from_rb(0.99) == pytest.approx(0.005)
    assert average_gate_fidelity_from_rb(0.99) == pytest.approx(0.995)


def test_gate_fidelity_perfect_when_p_one():
    assert average_gate_fidelity_from_rb(1.0) == pytest.approx(1.0)
    assert error_per_clifford_from_rb(1.0) == pytest.approx(0.0)


def test_gate_fidelity_dimension_factor():
    # d - 1 / d factor: for d = 4, r = (1 - p) * 3/4.
    p = 0.98
    assert error_per_clifford_from_rb(p, d=4) == pytest.approx((1 - p) * 0.75)
    assert average_gate_fidelity_from_rb(p, d=4) == pytest.approx(1 - (1 - p) * 0.75)


# --- coherence limit ------------------------------------------------------- #
def test_coherence_limit_matches_formula():
    # t_gate = 20 ns, T1 = 50 us, T2 = 30 us
    # e = (t/3)(1/T1 + 1/T2)
    t_gate, T1, T2 = 20e-9, 50e-6, 30e-6
    expected = (t_gate / 3.0) * (1.0 / T1 + 1.0 / T2)
    res = coherence_limit_error(t_gate, T1, T2)
    assert res.error == pytest.approx(expected)
    assert res.fidelity == pytest.approx(1.0 - expected)


def test_coherence_limit_zero_duration_is_lossless():
    res = coherence_limit_error(0.0, 50e-6, 30e-6)
    assert res.error == pytest.approx(0.0)
    assert res.fidelity == pytest.approx(1.0)


def test_coherence_limit_rejects_nonpositive_times():
    with pytest.raises(ValueError):
        coherence_limit_error(20e-9, 0.0, 30e-6)
    with pytest.raises(ValueError):
        coherence_limit_error(20e-9, 50e-6, -1.0)


# --- error budget ---------------------------------------------------------- #
def test_error_budget_excess_zero_when_rb_equals_coherence_limit():
    t_gate, T1, T2 = 20e-9, 50e-6, 30e-6
    coh = coherence_limit_error(t_gate, T1, T2).error
    # Choose rb_p so the RB error per Clifford exactly equals the coherence
    # limit: r = (1 - p)/2 = coh  ->  p = 1 - 2*coh.
    rb_p = 1.0 - 2.0 * coh
    budget = error_budget(t_gate, T1, T2, rb_p)
    assert budget.coherence_error == pytest.approx(coh)
    assert budget.measured_error == pytest.approx(coh)
    assert budget.excess_error == pytest.approx(0.0, abs=1e-12)


def test_error_budget_excess_positive_when_rb_exceeds_limit():
    t_gate, T1, T2 = 20e-9, 50e-6, 30e-6
    # A clearly control-limited gate: measured error well above the floor.
    budget = error_budget(t_gate, T1, T2, rb_p=0.99)
    assert budget.measured_error > budget.coherence_error
    assert budget.excess_error > 0.0
    assert budget.excess_error == pytest.approx(
        budget.measured_error - budget.coherence_error
    )


def test_error_budget_fidelities_consistent():
    budget = error_budget(20e-9, 50e-6, 30e-6, rb_p=0.99)
    assert budget.coherence_fidelity == pytest.approx(1.0 - budget.coherence_error)
    assert budget.measured_fidelity == pytest.approx(1.0 - budget.measured_error)
