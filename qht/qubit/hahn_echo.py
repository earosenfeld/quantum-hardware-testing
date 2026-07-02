"""Hahn-echo T2 experiment: refocusing low-frequency dephasing.

Protocol: pi/2 -- t/2 -- pi (refocus) -- t/2 -- pi/2 -- read out. The central
pi pulse reverses the phase accumulated from *static* (low-frequency) frequency
offsets, so quasi-static noise that limits the Ramsey T2* is cancelled and the
echo coherence time satisfies ``T2_echo > T2*`` for the same noise.

To make this emerge from physics rather than be asserted, the simulator carries
an explicit ensemble of quasi-static detunings (shot-to-shot frequency jitter,
std ``sigma_f``) plus a fast intrinsic dephasing rate ``1/T2_intrinsic``:

- Ramsey: each shot precesses at its own detuning; averaging over the Gaussian
  ensemble gives a Gaussian envelope exp(-(t/T2*_inhom)^2) on top of the
  intrinsic exponential. The inhomogeneous part dominates and sets a short T2*.
- Echo: the pi pulse cancels the static detuning of every ensemble member, so
  the inhomogeneous envelope disappears and only the intrinsic (long) decay
  remains -- hence a longer fitted T2_echo.

Both are fit with the same exponential ``A*exp(-t/T2)+C`` decay model, and the
test checks ``T2_echo > T2*`` from the recovered numbers.
"""

from __future__ import annotations

import numpy as np

from .models import exp_decay, fit_curve, sample_shots
from .relaxation import T1Result


def _ensemble_envelope(
    t: np.ndarray, sigma_f: float, t2_intrinsic: float, echo: bool
) -> np.ndarray:
    """Coherence-decay envelope C(t) in [0, 1] for Ramsey vs echo.

    The static-detuning ensemble (Gaussian, std ``sigma_f`` in Hz) contributes
    a Gaussian decay ``exp(-(2*pi*sigma_f*t)^2 / 2)`` to the *free-induction*
    (Ramsey) signal. A perfect refocusing pulse removes that term entirely for
    the echo. Both keep the intrinsic exponential ``exp(-t/T2_intrinsic)``.
    """
    intrinsic = np.exp(-t / t2_intrinsic)
    if echo:
        return intrinsic
    inhom = np.exp(-0.5 * (2.0 * np.pi * sigma_f * t) ** 2)
    return intrinsic * inhom


def simulate_hahn_echo(
    sigma_f: float,
    t2_intrinsic: float,
    echo: bool,
    delays: np.ndarray | None = None,
    A: float = 0.49,
    C: float = 0.5,
    n_shots: int = 8192,
    n_points: int = 60,
    span: float = 3.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a shot-noisy coherence-decay curve (Ramsey or Hahn echo).

    Parameters
    ----------
    sigma_f:
        Std of the quasi-static detuning ensemble (Hz). This is the low-frequency
        noise the echo refocuses.
    t2_intrinsic:
        Intrinsic (irreducible) dephasing time (s) common to both sequences.
    echo:
        ``False`` for the free-induction (Ramsey-like) decay, ``True`` for the
        refocused Hahn echo.

    The decay axis defaults to ``span`` times the *Ramsey* 1/e time so both
    curves are captured on a comparable window.
    """
    rng = np.random.default_rng(seed)
    if delays is None:
        # Ramsey 1/e time from the Gaussian inhomogeneous envelope.
        t2_star_inhom = 1.0 / (np.sqrt(2.0) * np.pi * sigma_f) if sigma_f > 0 else t2_intrinsic
        t_ref = min(t2_star_inhom, t2_intrinsic)
        delays = np.linspace(0.0, span * t_ref, n_points)
    delays = np.asarray(delays, dtype=float)

    env = _ensemble_envelope(delays, sigma_f, t2_intrinsic, echo)
    # Magnitude of the coherence-decay signal (envelope of the oscillation),
    # which is what a T2 echo/Ramsey-envelope fit targets.
    p_true = A * env + C
    p_hat, sigma = sample_shots(p_true, n_shots, rng)
    return delays, p_hat, sigma


def fit_hahn_echo(
    delays: np.ndarray, p_hat: np.ndarray, sigma: np.ndarray | None = None
) -> T1Result:
    """Fit the coherence envelope to ``A*exp(-t/T2)+C``; return T2 + uncertainty.

    Reuses the exponential model/result container; ``T1Result.T1`` here holds the
    fitted T2 (the field is generic to a single decay constant).
    """
    delays = np.asarray(delays, dtype=float)
    p_hat = np.asarray(p_hat, dtype=float)

    c0 = float(np.min(p_hat))
    a0 = float(np.clip(p_hat[0] - c0, 1e-3, 1.0))
    target = c0 + a0 / np.e
    idx = int(np.argmin(np.abs(p_hat - target)))
    t2_0 = max(delays[idx], (delays[-1] - delays[0]) / 4.0, 1e-9)

    fit = fit_curve(
        exp_decay,
        delays,
        p_hat,
        p0=[a0, t2_0, c0],
        names=("A", "T2", "C"),
        sigma=sigma,
        bounds=([0.0, 1e-12, -0.5], [1.5, np.inf, 1.5]),
    )
    return T1Result(
        T1=fit.value("T2"),
        T1_err=fit.error("T2"),
        A=fit.value("A"),
        C=fit.value("C"),
        fit=fit,
    )
