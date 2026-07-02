"""Generate publication-quality figures of the single-qubit characterization bench.

Each figure shows the noisy single-shot DATA POINTS (markers) together with the
least-squares FITTED CURVE (line) and the extracted parameter +/- its
covariance-based uncertainty -- i.e. exactly the artefact a hardware reviewer
reads off a real characterization run.

Run from the package root so ``from qht.qubit import ...`` resolves::

    cd quantum-hardware-test && .venv/bin/python scripts/make_figures.py

Outputs PNGs to ``quantum-hardware-test/assets/``. Headless (Agg), fixed seeds,
so the figures are byte-stable across runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``from qht.qubit import ...`` resolve regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

# --- house style (verbatim) ------------------------------------------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#334155", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e2e8f0", "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.frameon": False, "lines.linewidth": 2.0,
})
PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]
# ---------------------------------------------------------------------------- #

from qht.qubit import (  # noqa: E402
    simulate_t1, fit_t1,
    simulate_ramsey, fit_ramsey,
    simulate_rabi, fit_rabi,
    simulate_readout_iq, assignment_fidelity,
    simulate_rb, fit_rb,
    error_budget,
)

ASSETS = REPO_ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# Injected ground-truth parameters (match the CLI demo defaults).
SEED = 7
T1_TRUE = 50e-6          # 50 us
T2_TRUE = 30e-6          # 30 us
DETUNING_TRUE = 0.5e6    # 0.5 MHz
RABI_TRUE = 10e6         # 10 MHz
READOUT_SNR = 6.0        # blob separation / sigma
RB_P = 0.995             # depolarizing parameter per Clifford
T_GATE = 20e-9           # single-Clifford gate duration (20 ns)

BOX = dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#cbd5e1")


def _annotate(ax, text, loc="upper right"):
    """Place a fit-result text box in a corner of ``ax``."""
    xy = {"upper right": (0.97, 0.95), "upper left": (0.03, 0.95),
          "lower right": (0.97, 0.05), "lower left": (0.03, 0.05)}[loc]
    ha = "right" if "right" in loc else "left"
    va = "top" if "upper" in loc else "bottom"
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=10, family="monospace", bbox=BOX)


def _data_kw(color):
    return dict(fmt="o", ms=4.5, color=color, mfc="white", mec=color,
                mew=1.2, ecolor=color, elinewidth=1.0, capsize=2,
                alpha=0.9, zorder=3, label="single-shot data")


# --------------------------------------------------------------------------- #
def fig_t1():
    d, p, s = simulate_t1(T1_TRUE, n_shots=2048, seed=SEED)
    r = fit_t1(d, p, s)
    t = np.linspace(d.min(), d.max(), 400)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.errorbar(d * 1e6, p, yerr=s, **_data_kw(PALETTE[0]))
    ax.plot(t * 1e6, r.fit.predict(t), color=PALETTE[1], zorder=4,
            label=r"fit  $A\,e^{-t/T_1}+C$")
    ax.set_xlabel(r"delay $t$  ($\mu$s)")
    ax.set_ylabel(r"excited-state population $P_1$")
    ax.set_title("T$_1$ energy relaxation")
    _annotate(ax, f"T1 = {r.t1_us:.1f} +/- {r.T1_err * 1e6:.1f} us")
    ax.legend(loc="lower left")
    _save(fig, "t1_relaxation.png")


def fig_ramsey():
    d, p, s = simulate_ramsey(T2_TRUE, DETUNING_TRUE, n_shots=4096, seed=SEED + 1)
    r = fit_ramsey(d, p, s)
    t = np.linspace(d.min(), d.max(), 800)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.errorbar(d * 1e6, p, yerr=s, **_data_kw(PALETTE[4]))
    ax.plot(t * 1e6, r.fit.predict(t), color=PALETTE[1], zorder=4,
            label=r"fit  $A\,e^{-t/T_2^*}\cos(2\pi\Delta f\,t)+C$")
    # decay envelope (dashed) makes T2* visible at a glance
    env = r.A * np.exp(-t / r.T2) + r.C
    ax.plot(t * 1e6, env, color=PALETTE[1], ls="--", lw=1.2, alpha=0.7,
            zorder=2, label=r"$\pm$ envelope ($T_2^*$)")
    ax.plot(t * 1e6, -r.A * np.exp(-t / r.T2) + r.C, color=PALETTE[1],
            ls="--", lw=1.2, alpha=0.7, zorder=2)
    ax.set_xlabel(r"free-precession delay $t$  ($\mu$s)")
    ax.set_ylabel(r"population $P_1$")
    ax.set_title("Ramsey $T_2^*$ fringe + detuning")
    _annotate(ax,
              f"T2* = {r.t2_us:.1f} +/- {r.T2_err * 1e6:.1f} us\n"
              f"df  = {r.delta_f / 1e3:.1f} +/- {r.delta_f_err / 1e3:.1f} kHz")
    ax.legend(loc="lower left", ncol=1)
    _save(fig, "ramsey_t2.png")


def fig_rabi():
    d, p, s = simulate_rabi(RABI_TRUE, n_shots=4096, seed=SEED + 2)
    r = fit_rabi(d, p, s)
    t = np.linspace(d.min(), d.max(), 600)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.errorbar(d * 1e9, p, yerr=s, **_data_kw(PALETTE[2]))
    ax.plot(t * 1e9, r.fit.predict(t), color=PALETTE[1], zorder=4,
            label=r"fit  $\frac{A}{2}(1-\cos 2\pi f_R t)+C$")
    # mark the pi-pulse (first population maximum)
    ax.axvline(r.t_pi * 1e9, color=PALETTE[3], ls="--", lw=1.4, zorder=2)
    ymax = r.fit.predict(np.array([r.t_pi]))[0]
    ax.plot([r.t_pi * 1e9], [ymax], marker="*", ms=14, color=PALETTE[3],
            mec="#334155", mew=0.6, zorder=5,
            label=rf"$\pi$-pulse  ({r.t_pi_ns:.1f} ns)")
    ax.set_xlabel(r"drive duration $\tau$  (ns)")
    ax.set_ylabel(r"population $P_1$")
    ax.set_title("Rabi calibration $\\rightarrow$ $\\pi$-pulse")
    _annotate(ax,
              f"f_Rabi = {r.f_rabi / 1e6:.3f} +/- {r.f_rabi_err / 1e6:.3f} MHz\n"
              f"t_pi   = {r.t_pi_ns:.2f} +/- {r.t_pi_err * 1e9:.2f} ns",
              loc="upper left")
    ax.legend(loc="lower right")
    _save(fig, "rabi_calibration.png")


def fig_readout():
    iq0, iq1 = simulate_readout_iq(READOUT_SNR, 1.0, n_shots=6000, seed=SEED + 5)
    r = assignment_fidelity(iq0, iq1)

    fig, (ax, axc) = plt.subplots(
        1, 2, figsize=(9.4, 4.4), gridspec_kw={"width_ratios": [1.55, 1.0]})

    # --- IQ scatter ---
    ax.scatter(iq0[:, 0], iq0[:, 1], s=6, color=PALETTE[0], alpha=0.25,
               edgecolors="none", label=r"prepared $|0\rangle$")
    ax.scatter(iq1[:, 0], iq1[:, 1], s=6, color=PALETTE[1], alpha=0.25,
               edgecolors="none", label=r"prepared $|1\rangle$")
    # decision boundary: perpendicular to the |0>->|1> axis at the threshold
    mu0, mu1 = iq0.mean(0), iq1.mean(0)
    axis = (mu1 - mu0) / np.linalg.norm(mu1 - mu0)
    perp = np.array([-axis[1], axis[0]])
    mid = mu0 + axis * (r.threshold - mu0 @ axis)
    span = np.linspace(-4, 4, 2)
    ax.plot(mid[0] + perp[0] * span, mid[1] + perp[1] * span,
            color="#334155", ls="--", lw=1.4, zorder=4, label="discriminator")
    ax.scatter(*mu0, marker="x", s=80, color="#1e3a8a", lw=2.2, zorder=5)
    ax.scatter(*mu1, marker="x", s=80, color="#7f1d1d", lw=2.2, zorder=5)
    ax.set_xlabel("I (arb.)")
    ax.set_ylabel("Q (arb.)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Single-shot readout  (F = {r.fidelity:.4f})")
    leg = ax.legend(loc="upper left", markerscale=2.0, fontsize=9)
    for h in leg.legend_handles:
        if hasattr(h, "set_alpha"):
            h.set_alpha(0.9)

    # --- confusion matrix panel ---
    axc.grid(False)
    im = axc.imshow(r.confusion, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    axc.set_xticks([0, 1], [r"called $|0\rangle$", r"called $|1\rangle$"])
    axc.set_yticks([0, 1], [r"prep $|0\rangle$", r"prep $|1\rangle$"])
    for i in range(2):
        for j in range(2):
            val = r.confusion[i, j]
            axc.text(j, i, f"{val:.3f}", ha="center", va="center",
                     color="white" if val > 0.5 else "#1e293b",
                     fontsize=13, fontweight="bold")
    axc.set_title("confusion matrix")
    fig.colorbar(im, ax=axc, fraction=0.046, pad=0.04, label="P(assigned)")
    _save(fig, "readout_fidelity.png")


def fig_summary():
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    (a_t1, a_ram), (a_rabi, a_rb) = axes

    # T1
    d, p, s = simulate_t1(T1_TRUE, n_shots=2048, seed=SEED)
    r = fit_t1(d, p, s)
    t = np.linspace(d.min(), d.max(), 400)
    a_t1.errorbar(d * 1e6, p, yerr=s, **_data_kw(PALETTE[0]))
    a_t1.plot(t * 1e6, r.fit.predict(t), color=PALETTE[1], zorder=4)
    a_t1.set_title("T$_1$ relaxation")
    a_t1.set_xlabel(r"delay ($\mu$s)"); a_t1.set_ylabel(r"$P_1$")
    _annotate(a_t1, f"T1 = {r.t1_us:.1f} +/- {r.T1_err * 1e6:.1f} us")

    # Ramsey
    d, p, s = simulate_ramsey(T2_TRUE, DETUNING_TRUE, n_shots=4096, seed=SEED + 1)
    r = fit_ramsey(d, p, s)
    t = np.linspace(d.min(), d.max(), 800)
    a_ram.errorbar(d * 1e6, p, yerr=s, **_data_kw(PALETTE[4]))
    a_ram.plot(t * 1e6, r.fit.predict(t), color=PALETTE[1], zorder=4)
    a_ram.set_title("Ramsey $T_2^*$")
    a_ram.set_xlabel(r"delay ($\mu$s)"); a_ram.set_ylabel(r"$P_1$")
    _annotate(a_ram,
              f"T2* = {r.t2_us:.1f} +/- {r.T2_err * 1e6:.1f} us\n"
              f"df  = {r.delta_f / 1e3:.1f} +/- {r.delta_f_err / 1e3:.1f} kHz")

    # Rabi
    d, p, s = simulate_rabi(RABI_TRUE, n_shots=4096, seed=SEED + 2)
    r = fit_rabi(d, p, s)
    t = np.linspace(d.min(), d.max(), 600)
    a_rabi.errorbar(d * 1e9, p, yerr=s, **_data_kw(PALETTE[2]))
    a_rabi.plot(t * 1e9, r.fit.predict(t), color=PALETTE[1], zorder=4)
    a_rabi.axvline(r.t_pi * 1e9, color=PALETTE[3], ls="--", lw=1.3)
    a_rabi.set_title(r"Rabi $\rightarrow$ $\pi$-pulse")
    a_rabi.set_xlabel(r"duration (ns)"); a_rabi.set_ylabel(r"$P_1$")
    _annotate(a_rabi, f"t_pi = {r.t_pi_ns:.1f} +/- {r.t_pi_err * 1e9:.1f} ns",
              loc="upper left")

    # Randomized benchmarking
    L, S, sg = simulate_rb(RB_P, n_sequences=40, n_shots=4096, seed=SEED + 6)
    rb = fit_rb(L, S, sg)
    mm = np.linspace(L.min(), L.max(), 400)
    a_rb.errorbar(L, S, yerr=sg, **_data_kw(PALETTE[5]))
    a_rb.plot(mm, rb.fit.predict(mm), color=PALETTE[1], zorder=4)
    a_rb.set_xscale("log")
    a_rb.set_title("Randomized benchmarking")
    a_rb.set_xlabel("sequence length $m$ (Cliffords)")
    a_rb.set_ylabel("survival $P(|0\\rangle)$")
    _annotate(a_rb,
              f"p   = {rb.p:.4f} +/- {rb.p_err:.4f}\n"
              f"EPC = {rb.epc:.2e}")

    fig.suptitle("Single-qubit characterization battery (simulated hardware)",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    _save(fig, "characterization_summary.png")


def fig_error_budget():
    """Coherence-limited vs RB-measured gate error, with the control excess on top.

    The measured error comes from the same RB run as the rest of the gallery; the
    coherence floor comes from the injected T1/T2 over a single-Clifford gate.
    """
    L, S, sg = simulate_rb(RB_P, n_sequences=40, n_shots=4096, seed=SEED + 6)
    rb = fit_rb(L, S, sg)
    b = error_budget(T_GATE, T1_TRUE, T2_TRUE, rb_p=rb.p)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    # Bar 1: the coherence floor (what T1/T2 alone force).
    # Bar 2: the same floor + the measured excess stacked on top == measured error.
    x = [0, 1]
    ax.bar(x[0], b.coherence_error, width=0.62, color=PALETTE[2],
           label="coherence-limited (T1/T2 floor)", zorder=3)
    ax.bar(x[1], b.coherence_error, width=0.62, color=PALETTE[2], zorder=3)
    ax.bar(x[1], b.excess_error, width=0.62, bottom=b.coherence_error,
           color=PALETTE[1], label="excess (control / leakage)", zorder=3)

    ax.set_xticks(x, ["coherence\nlimit", "RB-measured\nerror"])
    ax.set_ylabel("average gate error per Clifford")
    ax.set_title("Gate error budget vs coherence limit")
    ax.set_ylim(0, b.measured_error * 1.28)

    # Value labels.
    ax.text(x[0], b.coherence_error, f"  {b.coherence_error:.2e}",
            ha="center", va="bottom", fontsize=9, family="monospace")
    ax.text(x[1], b.measured_error, f"  {b.measured_error:.2e}",
            ha="center", va="bottom", fontsize=9, family="monospace")

    _annotate(
        ax,
        f"t_gate = {T_GATE * 1e9:.0f} ns\n"
        f"T1 = {T1_TRUE * 1e6:.0f} us  T2 = {T2_TRUE * 1e6:.0f} us\n"
        f"p  = {rb.p:.4f}\n"
        f"F_RB   = {1 - b.measured_error:.5f}\n"
        f"e_coh  = {b.coherence_error:.2e}\n"
        f"excess = {b.excess_error:.2e}",
        loc="upper left",
    )
    ax.legend(loc="upper right")
    _save(fig, "error_budget.png")


def _save(fig, name):
    path = ASSETS / name
    fig.savefig(path)
    plt.close(fig)
    kb = path.stat().st_size / 1024
    print(f"  wrote {name:32s} {kb:6.1f} KB")


def main():
    print(f"Generating characterization figures -> {ASSETS}")
    fig_t1()
    fig_ramsey()
    fig_rabi()
    fig_readout()
    fig_error_budget()
    fig_summary()
    print("done.")


if __name__ == "__main__":
    main()
