"""Ramsey T2* experiment: dephasing time + qubit-drive detuning.

Protocol: pi/2 -- wait ``t`` -- pi/2 -- read out. In the rotating frame of the
drive, a detuning ``delta_f`` between the drive and the qubit makes the Bloch
vector precess, producing fringes that decay with the (Gaussian-ish, here
modelled as exponential) inhomogeneous dephasing time T2*:

    P1(t) = A * exp(-t / T2) * cos(2*pi*delta_f*t + phi) + C

Fitting recovers BOTH the envelope time constant T2* and the detuning
``delta_f`` (the fringe frequency), each with a covariance-based error bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import FitResult, fit_curve, ramsey_decay, sample_shots


@dataclass
class RamseyResult:
    """Fitted Ramsey parameters with 1-sigma uncertainties."""

    T2: float
    T2_err: float
    delta_f: float
    delta_f_err: float
    A: float
    C: float
    phi: float
    fit: FitResult

    @property
    def t2_us(self) -> float:
        return self.T2 * 1e6


def _guess_detuning(t: np.ndarray, y: np.ndarray) -> float:
    """Estimate fringe frequency from the real FFT of the (de-meaned) signal."""
    y = y - np.mean(y)
    n = len(t)
    dt = float(np.mean(np.diff(t)))
    if dt <= 0 or n < 4:
        return 1.0 / (t[-1] - t[0] + 1e-12)
    freqs = np.fft.rfftfreq(n, d=dt)
    amp = np.abs(np.fft.rfft(y))
    amp[0] = 0.0  # ignore DC
    f0 = freqs[int(np.argmax(amp))]
    if f0 <= 0:
        f0 = 1.0 / (t[-1] - t[0] + 1e-12)
    return float(f0)


def simulate_ramsey(
    t2_true: float,
    delta_f_true: float,
    delays: np.ndarray | None = None,
    A: float = 0.48,
    C: float = 0.5,
    phi: float = 0.0,
    n_shots: int = 4096,
    n_points: int = 80,
    span: float = 3.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a shot-noisy Ramsey fringe.

    Defaults model an ideal pi/2-pi/2 sequence whose population swings about 0.5
    with ~half-amplitude contrast. ``delays`` defaults to span * T2* sampled
    finely enough to resolve several detuning fringes.
    """
    rng = np.random.default_rng(seed)
    if delays is None:
        # Nyquist-respecting sampling: enough points to resolve the fringe.
        t_max = span * t2_true
        n_fringe = max(1.0, delta_f_true * t_max)
        n_points = int(max(n_points, 10 * n_fringe))
        delays = np.linspace(0.0, t_max, n_points)
    delays = np.asarray(delays, dtype=float)
    p_true = ramsey_decay(delays, A, t2_true, delta_f_true, phi, C)
    p_hat, sigma = sample_shots(p_true, n_shots, rng)
    return delays, p_hat, sigma


def fit_ramsey(
    delays: np.ndarray, p_hat: np.ndarray, sigma: np.ndarray | None = None
) -> RamseyResult:
    """Fit the Ramsey model, returning T2* and detuning with uncertainties."""
    delays = np.asarray(delays, dtype=float)
    p_hat = np.asarray(p_hat, dtype=float)

    c0 = float(np.mean(p_hat))
    a0 = float(np.clip((np.max(p_hat) - np.min(p_hat)) / 2.0, 1e-3, 1.0))
    f0 = _guess_detuning(delays, p_hat)
    t2_0 = max((delays[-1] - delays[0]) / 2.0, 1e-9)

    fit = fit_curve(
        ramsey_decay,
        delays,
        p_hat,
        p0=[a0, t2_0, f0, 0.0, c0],
        names=("A", "T2", "delta_f", "phi", "C"),
        sigma=sigma,
        bounds=(
            [0.0, 1e-12, 0.0, -2.0 * np.pi, -0.5],
            [1.5, np.inf, np.inf, 2.0 * np.pi, 1.5],
        ),
    )
    return RamseyResult(
        T2=fit.value("T2"),
        T2_err=fit.error("T2"),
        delta_f=fit.value("delta_f"),
        delta_f_err=fit.error("delta_f"),
        A=fit.value("A"),
        C=fit.value("C"),
        phi=fit.value("phi"),
        fit=fit,
    )
