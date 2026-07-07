"""Generate assets/t1_live_fit.gif.

Animated T1 acquisition as a real bench runs it: delay points stream in one at
a time (shot-noise error bars from :func:`qht.qubit.relaxation.simulate_t1`),
and after each new point the exponential is refit with
:func:`qht.qubit.relaxation.fit_t1` — the running estimate and its
covariance-based 1-sigma bar converge onto the injected ground truth.

Run from the repo root:

    python scripts/make_demo_gif.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from qht.qubit.models import exp_decay
from qht.qubit.relaxation import fit_t1, simulate_t1

OUT = Path(__file__).resolve().parents[1] / "assets" / "t1_live_fit.gif"

BG = "#0d1117"
FG = "#e6edf3"
C_DATA = "#58a6ff"
C_FIT = "#f0883e"
C_TRUE = "#3fb950"

T1_TRUE = 75e-6  # 75 us
MIN_POINTS = 6


def main() -> None:
    delays, p_hat, sigma = simulate_t1(T1_TRUE, n_points=36, n_shots=1024, seed=7)

    # Precompute the running fit after each acquired point.
    fits = {}
    for k in range(MIN_POINTS, delays.size + 1):
        try:
            fits[k] = fit_t1(delays[:k], p_hat[:k], sigma[:k])
        except Exception:
            pass

    fig, (ax, ax_c) = plt.subplots(
        1, 2, figsize=(9.4, 4.2), facecolor=BG,
        gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.3},
    )
    for a in (ax, ax_c):
        a.set_facecolor("#161b22")
        for spine in a.spines.values():
            spine.set_color("#30363d")
        a.tick_params(colors=FG, labelsize=8)

    us = 1e6
    t_dense = np.linspace(0, delays[-1], 400)

    err = ax.errorbar([], [], yerr=[], fmt="o", ms=4, color=C_DATA,
                      ecolor=C_DATA, elinewidth=1, capsize=2)
    (fit_ln,) = ax.plot([], [], color=C_FIT, lw=1.8, label="running fit")
    ax.plot(t_dense * us, exp_decay(t_dense, 0.98, T1_TRUE, 0.02),
            color=C_TRUE, lw=1.0, ls="--", alpha=0.8,
            label=f"truth: T1 = {T1_TRUE*us:.0f} µs")
    txt = ax.text(0.97, 0.82, "", transform=ax.transAxes, ha="right",
                  color=C_FIT, fontsize=11, fontweight="bold")
    ax.set_xlim(0, delays[-1] * us)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("delay [µs]", color=FG, fontsize=9)
    ax.set_ylabel(r"$P(|1\rangle)$", color=FG, fontsize=9)
    ax.set_title("T1 relaxation: live acquisition + refit", color=FG, fontsize=10)
    ax.legend(loc="upper right", fontsize=8, facecolor="#161b22",
              edgecolor="#30363d", labelcolor=FG)

    ks = sorted(fits)
    est = np.array([fits[k].t1_us for k in ks])
    est_err = np.array([fits[k].T1_err * us for k in ks])
    (conv_ln,) = ax_c.plot([], [], color=C_FIT, lw=1.5)
    band = [None]
    ax_c.axhline(T1_TRUE * us, color=C_TRUE, ls="--", lw=1.0)
    ax_c.set_xlim(MIN_POINTS, delays.size)
    lo = min(est.min() - est_err.max(), T1_TRUE * us * 0.85)
    hi = max(est.max() + est_err.max(), T1_TRUE * us * 1.15)
    ax_c.set_ylim(lo, hi)
    ax_c.set_xlabel("points acquired", color=FG, fontsize=9)
    ax_c.set_ylabel("fitted T1 ± 1σ [µs]", color=FG, fontsize=9)
    ax_c.set_title("Estimate converges", color=FG, fontsize=10)

    def update(frame: int):
        k = frame + 1
        # Left: acquired points + running fit.
        xs, ys, ss = delays[:k] * us, p_hat[:k], sigma[:k]
        # Redraw errorbar containers (simplest reliable way).
        for artist in list(ax.containers):
            artist.remove()
        ax.errorbar(xs, ys, yerr=ss, fmt="o", ms=4, color=C_DATA,
                    ecolor=C_DATA, elinewidth=1, capsize=2)
        if k in fits:
            r = fits[k]
            fit_ln.set_data(t_dense * us, exp_decay(t_dense, r.A, r.T1, r.C))
            txt.set_text(f"T1 = {r.t1_us:.1f} ± {r.T1_err*us:.1f} µs")
            # Right: convergence trace with 1-sigma band.
            n_pts = ks.index(k) + 1
            conv_ln.set_data(ks[:n_pts], est[:n_pts])
            if band[0] is not None:
                band[0].remove()
            band[0] = ax_c.fill_between(
                ks[:n_pts], est[:n_pts] - est_err[:n_pts],
                est[:n_pts] + est_err[:n_pts], color=C_FIT, alpha=0.18, lw=0)
        return []

    anim = FuncAnimation(fig, update, frames=delays.size, blit=False)
    OUT.parent.mkdir(exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=6), dpi=85)
    final = fits[delays.size]
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB); "
          f"final T1 = {final.t1_us:.2f} ± {final.T1_err*1e6:.2f} µs")


if __name__ == "__main__":
    main()
