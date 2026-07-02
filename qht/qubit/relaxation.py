"""T1 energy-relaxation experiment: simulate, then fit with uncertainty.

Protocol: prepare the qubit in |1> (a pi pulse), wait a variable delay ``t``,
then read out. The excited-state population decays exponentially toward its
steady state:

    P1(t) = A * exp(-t / T1) + C

with ``A ~ 1`` (initial excited population, less any readout/prep error) and a
small offset ``C`` (thermal equilibrium population + readout bias). Each delay
is measured with ``n_shots`` single-shot readouts, so the estimate carries
binomial shot noise -- exactly what is fit on real hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import FitResult, exp_decay, fit_curve, sample_shots


@dataclass
class T1Result:
    """Fitted T1 with its 1-sigma uncertainty (in seconds)."""

    T1: float
    T1_err: float
    A: float
    C: float
    fit: FitResult

    @property
    def t1_us(self) -> float:
        return self.T1 * 1e6


def t1_delays(t1_true: float, n_points: int = 40, span: float = 4.0) -> np.ndarray:
    """A sensible delay sweep: 0 out to ``span`` * T1 (a few e-foldings)."""
    return np.linspace(0.0, span * t1_true, n_points)


def simulate_t1(
    t1_true: float,
    delays: np.ndarray | None = None,
    A: float = 0.98,
    C: float = 0.02,
    n_shots: int = 2048,
    n_points: int = 40,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a shot-noisy T1 relaxation curve.

    Parameters
    ----------
    t1_true:
        Injected relaxation time (s).
    delays:
        Delay points (s). If ``None`` a default linear sweep is used.
    A, C:
        Contrast and offset of the ideal curve. Defaults model ~98% prep/readout
        contrast with a 2% residual excited population.
    n_shots:
        Single-shot measurements averaged per delay (sets the noise floor).

    Returns
    -------
    (delays, p_hat, sigma):
        Delay axis, measured populations, per-point binomial std error.
    """
    rng = np.random.default_rng(seed)
    if delays is None:
        delays = t1_delays(t1_true, n_points=n_points)
    delays = np.asarray(delays, dtype=float)
    p_true = exp_decay(delays, A, t1_true, C)
    p_hat, sigma = sample_shots(p_true, n_shots, rng)
    return delays, p_hat, sigma


def fit_t1(
    delays: np.ndarray, p_hat: np.ndarray, sigma: np.ndarray | None = None
) -> T1Result:
    """Fit ``A*exp(-t/T1)+C`` and return T1 with covariance-based uncertainty."""
    delays = np.asarray(delays, dtype=float)
    p_hat = np.asarray(p_hat, dtype=float)

    # Robust initial guesses from the data shape.
    c0 = float(np.min(p_hat))
    a0 = float(np.clip(p_hat[0] - c0, 1e-3, 1.0))
    # Estimate T1 from where the curve falls to 1/e of its initial contrast.
    target = c0 + a0 / np.e
    idx = int(np.argmin(np.abs(p_hat - target)))
    t1_0 = max(delays[idx], (delays[-1] - delays[0]) / 4.0, 1e-9)

    fit = fit_curve(
        exp_decay,
        delays,
        p_hat,
        p0=[a0, t1_0, c0],
        names=("A", "T1", "C"),
        sigma=sigma,
        bounds=([0.0, 1e-12, -0.5], [1.5, np.inf, 1.0]),
    )
    return T1Result(
        T1=fit.value("T1"),
        T1_err=fit.error("T1"),
        A=fit.value("A"),
        C=fit.value("C"),
        fit=fit,
    )
