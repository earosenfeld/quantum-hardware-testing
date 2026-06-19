"""Single-qubit characterization layer for the quantum hardware test bench.

This package provides closed-form simulators and least-squares fitters for the
standard single-qubit characterization experiments run on superconducting /
trapped-ion hardware inside a dilution refrigerator:

- T1 energy relaxation (``relaxation``)
- T2* Ramsey free-induction decay with detuning (``ramsey``)
- Rabi amplitude/duration calibration -> pi-pulse (``rabi``)
- Hahn-echo T2 with a refocusing pulse (``hahn_echo``)
- Single-shot dispersive readout assignment fidelity (``readout``)

All simulators inject realistic readout shot noise (binomial sampling of N
single-shot measurements per point), and all fitters report parameter
uncertainties derived from the covariance matrix returned by
``scipy.optimize.curve_fit`` (``perr = sqrt(diag(pcov))``).

An optional QuTiP Lindblad-master-equation engine (``lindblad``) lets T1/T2
emerge from open-system dynamics; it is imported lazily so the rest of the
package works with only numpy/scipy installed.
"""

from .models import (
    FitResult,
    exp_decay,
    ramsey_decay,
    rabi_cosine,
    fit_curve,
)
from .relaxation import simulate_t1, fit_t1, T1Result
from .ramsey import simulate_ramsey, fit_ramsey, RamseyResult
from .rabi import simulate_rabi, fit_rabi, RabiResult
from .hahn_echo import simulate_hahn_echo, fit_hahn_echo
from .readout import simulate_readout_iq, assignment_fidelity, ReadoutFidelityResult
from .randomized_benchmarking import (
    simulate_rb,
    fit_rb,
    RBResult,
    epc_from_p,
)

__all__ = [
    "FitResult",
    "exp_decay",
    "ramsey_decay",
    "rabi_cosine",
    "fit_curve",
    "simulate_t1",
    "fit_t1",
    "T1Result",
    "simulate_ramsey",
    "fit_ramsey",
    "RamseyResult",
    "simulate_rabi",
    "fit_rabi",
    "RabiResult",
    "simulate_hahn_echo",
    "fit_hahn_echo",
    "simulate_readout_iq",
    "assignment_fidelity",
    "ReadoutFidelityResult",
    "simulate_rb",
    "fit_rb",
    "RBResult",
    "epc_from_p",
]
