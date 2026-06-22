"""Gate fidelity and the coherence-limited error budget.

This module connects the bench's primitive measurements -- randomized
benchmarking (RB) and the coherence times T1 / T2 from relaxation, Ramsey and
Hahn-echo -- into the two numbers a hardware team actually reports for a gate:
its **average gate fidelity** and its **coherence-limited error budget**.

Average gate fidelity from RB
-----------------------------
RB returns a single depolarizing parameter ``p`` (the decay base of the
``A*p^m+B`` survival curve, SPAM-free). The average error per Clifford is

    r = (1 - p) * (d - 1) / d,

and the average gate fidelity is

    F = 1 - r.

Here ``d = 2 ** n`` is the Hilbert-space dimension for ``n`` qubits; for a
single qubit (``n = 1``) ``d = 2`` and ``r = (1 - p) / 2``. (This ``r`` is the
RB error per *Clifford*; an average physical gate involves ~1.5 generators per
Clifford on many architectures, but RB reports the per-Clifford figure and that
is what we compare against, so no generator rescaling is applied here.)

Coherence limit
---------------
Even a perfect control pulse loses fidelity simply because the qubit relaxes
(T1) and dephases (T2) while the gate executes. For a gate of duration
``t_gate`` the standard small-error coherence limit on the *average* gate error
of a single qubit is

    e_coh ~= (t_gate / 3) * (1 / T1 + 1 / T2),   F_coh = 1 - e_coh.

This is the leading-order ``t_gate << T1, T2`` expansion of the average gate
fidelity of the amplitude+phase-damping channel acting for time ``t_gate``; the
1/3 is the single-qubit (d = 2) Haar average over input states. ``T2`` here is
the total transverse decay time (``1/T2 = 1/(2 T1) + 1/T_phi``), so passing the
Hahn-echo or Ramsey T2 is appropriate.

Error budget
------------
Comparing the RB-measured error to this floor decomposes the measured error
into the part forced by decoherence and the **excess** -- everything else
(control imperfection, leakage, calibration drift):

    excess = e_measured - e_coh.

A positive excess is the headroom a better pulse could recover; an excess near
zero means the gate is already coherence-limited and only longer T1/T2 will
help. (Statistical scatter can make a measured excess slightly negative; that
just means the gate is at its coherence floor within error bars.)

All quantities are dimensionless errors/fidelities except ``t_gate``, ``T1`` and
``T2``, which share consistent time units (seconds in the bench's convention).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoherenceLimit:
    """Coherence-limited average gate error and the fidelity it implies."""

    error: float
    fidelity: float


@dataclass
class ErrorBudget:
    """Decomposition of a gate's measured error against its coherence floor.

    Attributes
    ----------
    coherence_error:
        Average gate error forced by T1/T2 decay over ``t_gate``.
    measured_error:
        Average error per Clifford from RB (``r = (1 - p)(d - 1)/d``).
    excess_error:
        ``measured_error - coherence_error`` -- the control/leakage budget.
    coherence_fidelity, measured_fidelity:
        ``1 - error`` for the two errors above (convenience).
    """

    coherence_error: float
    measured_error: float
    excess_error: float
    coherence_fidelity: float
    measured_fidelity: float


def average_gate_fidelity_from_rb(p: float, d: int = 2) -> float:
    """Average gate fidelity from the RB depolarizing parameter ``p``.

    error per Clifford ``r = (1 - p) * (d - 1) / d`` and ``F = 1 - r``.
    For a single qubit ``d = 2`` so ``r = (1 - p) / 2``.

    Parameters
    ----------
    p:
        Depolarizing parameter (decay base) from the RB fit; ``p = 1`` is
        error-free.
    d:
        Hilbert-space dimension (``2`` for one qubit, ``2 ** n`` for ``n``).
    """
    r = error_per_clifford_from_rb(p, d)
    return 1.0 - r


def error_per_clifford_from_rb(p: float, d: int = 2) -> float:
    """RB error per Clifford ``r = (1 - p) * (d - 1) / d`` (``d = 2`` single qubit)."""
    return (1.0 - p) * (d - 1) / d


def coherence_limit_error(t_gate: float, T1: float, T2: float) -> CoherenceLimit:
    """Coherence-limited average gate error for a single-qubit gate.

    ``e_coh ~= (t_gate / 3) * (1 / T1 + 1 / T2)`` and ``F_coh = 1 - e_coh``
    (leading-order ``t_gate << T1, T2`` limit; the 1/3 is the d = 2 Haar
    average). ``T2`` is the total transverse decay time, so the Ramsey or
    Hahn-echo T2 may be passed directly.

    Parameters
    ----------
    t_gate:
        Gate duration (same time unit as ``T1``/``T2``).
    T1, T2:
        Relaxation and (total) transverse coherence times; must be > 0.

    Returns
    -------
    CoherenceLimit with ``.error`` and ``.fidelity``.
    """
    if T1 <= 0 or T2 <= 0:
        raise ValueError("T1 and T2 must be positive")
    if t_gate < 0:
        raise ValueError("t_gate must be non-negative")
    error = (t_gate / 3.0) * (1.0 / T1 + 1.0 / T2)
    return CoherenceLimit(error=error, fidelity=1.0 - error)


def error_budget(
    t_gate: float, T1: float, T2: float, rb_p: float, d: int = 2
) -> ErrorBudget:
    """Compare an RB-measured gate error to its coherence limit.

    Computes the coherence-limited error (from ``t_gate``, ``T1``, ``T2``) and
    the RB-measured error per Clifford (from ``rb_p``), and reports their
    difference as the **excess** (control/leakage) error::

        excess = measured_error - coherence_error.

    Parameters
    ----------
    t_gate, T1, T2:
        Gate duration and coherence times (consistent units) for the floor.
    rb_p:
        RB depolarizing parameter for the measured error.
    d:
        Hilbert-space dimension (``2`` single qubit).

    Returns
    -------
    ErrorBudget with the coherence-limited, measured and excess errors (plus the
    two implied fidelities).
    """
    coh = coherence_limit_error(t_gate, T1, T2)
    measured = error_per_clifford_from_rb(rb_p, d)
    excess = measured - coh.error
    return ErrorBudget(
        coherence_error=coh.error,
        measured_error=measured,
        excess_error=excess,
        coherence_fidelity=coh.fidelity,
        measured_fidelity=1.0 - measured,
    )
