"""Rabi calibration: drive-duration sweep -> Rabi frequency -> pi-pulse.

Protocol: drive the qubit on resonance for a variable duration ``tau`` and read
out. The excited population oscillates at the Rabi frequency set by the drive
amplitude:

    P1(tau) = (A / 2) * (1 - cos(2*pi*f_rabi*tau)) + C

This is the gate-calibration primitive: the pi pulse (full population transfer)
has duration ``t_pi = 1 / (2 * f_rabi)``, the first maximum of the curve. The
fitter therefore reports ``f_rabi`` and the derived ``t_pi`` (with its
uncertainty propagated from ``f_rabi``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import FitResult, fit_curve, rabi_cosine, sample_shots


@dataclass
class RabiResult:
    """Fitted Rabi calibration with uncertainties (durations in seconds)."""

    f_rabi: float
    f_rabi_err: float
    t_pi: float
    t_pi_err: float
    A: float
    C: float
    fit: FitResult

    @property
    def t_pi_ns(self) -> float:
        return self.t_pi * 1e9


def _guess_rabi_freq(tau: np.ndarray, y: np.ndarray) -> float:
    y = y - np.mean(y)
    n = len(tau)
    dt = float(np.mean(np.diff(tau)))
    if dt <= 0 or n < 4:
        return 1.0 / (tau[-1] - tau[0] + 1e-12)
    freqs = np.fft.rfftfreq(n, d=dt)
    amp = np.abs(np.fft.rfft(y))
    amp[0] = 0.0
    f0 = freqs[int(np.argmax(amp))]
    if f0 <= 0:
        f0 = 1.0 / (tau[-1] - tau[0] + 1e-12)
    return float(f0)


def simulate_rabi(
    f_rabi_true: float,
    durations: np.ndarray | None = None,
    A: float = 0.97,
    C: float = 0.015,
    n_shots: int = 4096,
    n_periods: float = 3.0,
    n_points: int = 80,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a shot-noisy Rabi oscillation.

    By default the duration axis spans ``n_periods`` full Rabi cycles so the
    oscillation (and therefore the pi-pulse point) is well constrained.
    """
    rng = np.random.default_rng(seed)
    if durations is None:
        t_max = n_periods / f_rabi_true
        durations = np.linspace(0.0, t_max, n_points)
    durations = np.asarray(durations, dtype=float)
    p_true = rabi_cosine(durations, A, f_rabi_true, C)
    p_hat, sigma = sample_shots(p_true, n_shots, rng)
    return durations, p_hat, sigma


def fit_rabi(
    durations: np.ndarray, p_hat: np.ndarray, sigma: np.ndarray | None = None
) -> RabiResult:
    """Fit the Rabi model and derive the pi-pulse duration with uncertainty."""
    durations = np.asarray(durations, dtype=float)
    p_hat = np.asarray(p_hat, dtype=float)

    c0 = float(np.min(p_hat))
    a0 = float(np.clip(np.max(p_hat) - np.min(p_hat), 1e-3, 1.0))
    f0 = _guess_rabi_freq(durations, p_hat)

    fit = fit_curve(
        rabi_cosine,
        durations,
        p_hat,
        p0=[a0, f0, c0],
        names=("A", "f_rabi", "C"),
        sigma=sigma,
        bounds=([0.0, 0.0, -0.5], [1.5, np.inf, 1.0]),
    )

    f_rabi = fit.value("f_rabi")
    f_rabi_err = fit.error("f_rabi")
    t_pi = 1.0 / (2.0 * f_rabi)
    # Propagate uncertainty: t_pi = 1/(2 f)  =>  d t_pi = t_pi * (df / f).
    t_pi_err = t_pi * (f_rabi_err / f_rabi) if f_rabi > 0 else float("inf")

    return RabiResult(
        f_rabi=f_rabi,
        f_rabi_err=f_rabi_err,
        t_pi=t_pi,
        t_pi_err=t_pi_err,
        A=fit.value("A"),
        C=fit.value("C"),
        fit=fit,
    )
