"""Fit models and a thin least-squares wrapper shared by the experiments.

Every characterization fitter in this package boils down to a nonlinear
least-squares fit of one of the closed-form population models below to a set
of noisy ``P1`` estimates. ``fit_curve`` wraps :func:`scipy.optimize.curve_fit`
and always propagates the parameter uncertainty from the covariance matrix
(``perr = sqrt(diag(pcov))``), which is the number a hardware reviewer cares
about: the reported time constant is meaningless without its error bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import curve_fit


# --------------------------------------------------------------------------- #
# Closed-form population models. ``t``/``tau`` are in seconds, rates in 1/s,
# frequencies in Hz. ``P1`` is the excited-state population in [0, 1].
# --------------------------------------------------------------------------- #
def exp_decay(t: np.ndarray, A: float, T1: float, C: float) -> np.ndarray:
    """T1 relaxation: ``P1(t) = A * exp(-t / T1) + C``."""
    return A * np.exp(-t / T1) + C


def ramsey_decay(
    t: np.ndarray, A: float, T2: float, delta_f: float, phi: float, C: float
) -> np.ndarray:
    """Ramsey free-induction decay.

    ``P1(t) = A * exp(-t / T2) * cos(2*pi*delta_f*t + phi) + C``
    """
    return A * np.exp(-t / T2) * np.cos(2.0 * np.pi * delta_f * t + phi) + C


def rabi_cosine(tau: np.ndarray, A: float, f_rabi: float, C: float) -> np.ndarray:
    """Rabi oscillation vs drive duration.

    ``P1(tau) = (A / 2) * (1 - cos(2*pi*f_rabi*tau)) + C``

    With ``C = 0`` and ``A = 1`` this runs from 0 (identity) to 1 (pi pulse).
    """
    return (A / 2.0) * (1.0 - np.cos(2.0 * np.pi * f_rabi * tau)) + C


@dataclass
class FitResult:
    """Container for a least-squares fit outcome.

    Attributes
    ----------
    names:
        Ordered parameter names.
    popt:
        Best-fit parameter values.
    perr:
        1-sigma uncertainties, ``sqrt(diag(pcov))``.
    pcov:
        Full covariance matrix.
    model:
        The fitted model callable (``model(x, *popt)``).
    """

    names: Sequence[str]
    popt: np.ndarray
    perr: np.ndarray
    pcov: np.ndarray
    model: Callable[..., np.ndarray] = field(repr=False)

    def value(self, name: str) -> float:
        return float(self.popt[list(self.names).index(name)])

    def error(self, name: str) -> float:
        return float(self.perr[list(self.names).index(name)])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model(np.asarray(x, dtype=float), *self.popt)

    def as_dict(self) -> dict:
        return {
            n: {"value": self.value(n), "stderr": self.error(n)} for n in self.names
        }


def fit_curve(
    model: Callable[..., np.ndarray],
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: Sequence[float],
    names: Sequence[str],
    sigma: np.ndarray | None = None,
    bounds: tuple | None = None,
    maxfev: int = 200_000,
) -> FitResult:
    """Least-squares fit with covariance-based uncertainties.

    Parameters
    ----------
    model:
        Model function ``f(x, *params)``.
    xdata, ydata:
        Independent variable and measured values.
    p0:
        Initial parameter guess (order must match ``names``).
    names:
        Parameter names, used for labelled lookup on the result.
    sigma:
        Optional per-point 1-sigma measurement errors. When provided the fit
        is properly weighted and ``absolute_sigma=True`` so the covariance is
        in physical units rather than rescaled by the reduced chi-squared.
    bounds:
        Optional ``(lower, upper)`` bounds passed to ``curve_fit``.

    Returns
    -------
    FitResult
    """
    xdata = np.asarray(xdata, dtype=float)
    ydata = np.asarray(ydata, dtype=float)

    kwargs: dict = {"p0": list(p0), "maxfev": maxfev}
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        # Guard against zero-variance points (e.g. P1 estimate of exactly 0/1):
        # a zero sigma would make curve_fit divide by zero. Floor it at one
        # shot's worth of binomial spread so those points are merely
        # down-weighted, not fatal.
        sigma = np.where(sigma <= 0, np.nanmin(sigma[sigma > 0], initial=1e-6), sigma)
        kwargs["sigma"] = sigma
        kwargs["absolute_sigma"] = True
    if bounds is not None:
        kwargs["bounds"] = bounds

    popt, pcov = curve_fit(model, xdata, ydata, **kwargs)
    perr = np.sqrt(np.diag(pcov))
    return FitResult(names=tuple(names), popt=popt, perr=perr, pcov=pcov, model=model)


# --------------------------------------------------------------------------- #
# Shot-noise helpers
# --------------------------------------------------------------------------- #
def sample_shots(
    p1: np.ndarray, n_shots: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample binomial single-shot readout for a vector of true ``P1`` values.

    For each ideal population ``p`` we draw ``k ~ Binomial(n_shots, p)`` excited
    counts and return the empirical estimate ``k / n_shots`` together with its
    binomial standard error ``sqrt(p_hat*(1-p_hat)/n_shots)`` for use as the
    fit weights.

    Returns
    -------
    (p_hat, sigma):
        Estimated populations and their per-point standard errors.
    """
    p1 = np.clip(np.asarray(p1, dtype=float), 0.0, 1.0)
    counts = rng.binomial(n_shots, p1)
    p_hat = counts / n_shots
    # Standard error of a binomial proportion. Floor at the n_shots=1 single
    # "phantom count" level so estimates that land on 0 or 1 still carry a
    # finite, sensible error bar instead of zero.
    var = p_hat * (1.0 - p_hat) / n_shots
    floor = (0.5 / n_shots) ** 2
    sigma = np.sqrt(np.maximum(var, floor))
    return p_hat, sigma
