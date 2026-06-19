"""Single-shot dispersive readout assignment fidelity in the IQ plane.

Dispersive readout maps the qubit state onto a coherent state of the readout
resonator; after demodulation each single shot is a point in the IQ plane.
Prepared |0> and |1> appear as two 2D Gaussian blobs. Their separation over the
blob width sets the measurement SNR. We:

1. Simulate ``n_shots`` IQ points for each prepared state.
2. Choose the optimal linear (projection + threshold) discriminator along the
   axis joining the two blob centres -- the standard Gaussian-blob classifier.
3. Build the 2x2 confusion matrix and report the assignment fidelity

       F = 1 - 0.5 * (P(1|0) + P(0|1))

   together with the per-state assignment fidelities P(0|0) and P(1|1).

High SNR -> blobs separate -> F -> 1. Overlapping blobs -> F -> 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReadoutFidelityResult:
    """Readout characterization summary."""

    fidelity: float  # F = 1 - 0.5*(P(1|0)+P(0|1))
    f0: float  # P(0|0): correct assignment of prepared |0>
    f1: float  # P(1|1): correct assignment of prepared |1>
    confusion: np.ndarray  # 2x2, rows = prepared, cols = assigned
    snr: float  # |centre separation| / sigma
    threshold: float  # decision threshold along the projection axis

    @property
    def p1_given_0(self) -> float:
        return float(self.confusion[0, 1])

    @property
    def p0_given_1(self) -> float:
        return float(self.confusion[1, 0])


def simulate_readout_iq(
    separation: float,
    sigma: float,
    n_shots: int = 5000,
    center0: tuple[float, float] = (0.0, 0.0),
    angle: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate IQ-plane single shots for prepared |0> and |1>.

    Parameters
    ----------
    separation:
        Distance between the |0> and |1> blob centres (arb. IQ units).
    sigma:
        Isotropic Gaussian blob width (same units). SNR = separation / sigma.
    angle:
        Orientation of the |0>->|1> axis in the IQ plane (radians); the
        discriminator must work at any angle, so tests exercise non-zero values.

    Returns
    -------
    (iq0, iq1):
        Arrays of shape ``(n_shots, 2)`` of IQ points for each prepared state.
    """
    rng = np.random.default_rng(seed)
    c0 = np.asarray(center0, dtype=float)
    direction = np.array([np.cos(angle), np.sin(angle)])
    c1 = c0 + separation * direction

    iq0 = rng.normal(loc=c0, scale=sigma, size=(n_shots, 2))
    iq1 = rng.normal(loc=c1, scale=sigma, size=(n_shots, 2))
    return iq0, iq1


def assignment_fidelity(iq0: np.ndarray, iq1: np.ndarray) -> ReadoutFidelityResult:
    """Classify IQ shots with an optimal linear discriminator and score it.

    The decision axis is the line joining the two empirical blob means; each
    shot is projected onto it and thresholded at the midpoint of the projected
    means (the Bayes-optimal threshold for equal-variance, equal-prior Gaussian
    blobs).
    """
    iq0 = np.asarray(iq0, dtype=float)
    iq1 = np.asarray(iq1, dtype=float)

    mu0 = iq0.mean(axis=0)
    mu1 = iq1.mean(axis=0)
    axis = mu1 - mu0
    norm = np.linalg.norm(axis)
    if norm == 0:
        # Degenerate (identical means): no information, assign everything to |0>.
        axis = np.array([1.0, 0.0])
        norm = 1.0
    axis = axis / norm

    proj0 = iq0 @ axis
    proj1 = iq1 @ axis
    threshold = 0.5 * (proj0.mean() + proj1.mean())

    # Assign |1> when the projection sits past the threshold on the |1> side.
    assign0 = (proj0 > threshold).astype(int)  # 1 => misassigned |0> as |1>
    assign1 = (proj1 > threshold).astype(int)  # 1 => correctly called |1>

    p1_given_0 = float(assign0.mean())
    p0_given_0 = 1.0 - p1_given_0
    p1_given_1 = float(assign1.mean())
    p0_given_1 = 1.0 - p1_given_1

    confusion = np.array(
        [[p0_given_0, p1_given_0], [p0_given_1, p1_given_1]], dtype=float
    )
    fidelity = 1.0 - 0.5 * (p1_given_0 + p0_given_1)

    # SNR from the empirical geometry.
    pooled_sigma = float(np.sqrt(0.5 * (proj0.var() + proj1.var())))
    snr = float(np.linalg.norm(mu1 - mu0) / pooled_sigma) if pooled_sigma > 0 else np.inf

    return ReadoutFidelityResult(
        fidelity=fidelity,
        f0=p0_given_0,
        f1=p1_given_1,
        confusion=confusion,
        snr=snr,
        threshold=threshold,
    )
