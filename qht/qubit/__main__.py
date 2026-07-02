"""CLI entry point for the single-qubit characterization bench.

Runs the standard characterization battery against the built-in simulators
(injecting known parameters + shot noise) and prints the fitted values with
their covariance-based uncertainties -- a self-contained demo of the bench
without any hardware attached.

    qubit-characterize                 # default injected parameters
    qubit-characterize --shots 8192    # tighter error bars
    python -m qht.qubit --t1-us 75
"""

from __future__ import annotations

import argparse

from .relaxation import simulate_t1, fit_t1
from .ramsey import simulate_ramsey, fit_ramsey
from .rabi import simulate_rabi, fit_rabi
from .readout import simulate_readout_iq, assignment_fidelity
from .hahn_echo import simulate_hahn_echo, fit_hahn_echo
from .randomized_benchmarking import simulate_rb, fit_rb
from .lindblad import have_qutip


def _fmt(value: float, err: float, scale: float, unit: str) -> str:
    return f"{value * scale:8.3f} +/- {err * scale:6.3f} {unit}"


def run_battery(args: argparse.Namespace) -> None:
    seed = args.seed
    shots = args.shots

    print("Single-qubit characterization bench (simulated hardware)")
    print("=" * 60)

    # T1
    t1_true = args.t1_us * 1e-6
    d, p, s = simulate_t1(t1_true, n_shots=shots, seed=seed)
    r = fit_t1(d, p, s)
    print(f"T1 relaxation : {_fmt(r.T1, r.T1_err, 1e6, 'us')}   (injected {t1_true*1e6:.1f} us)")

    # Ramsey
    t2_true, df_true = args.t2_us * 1e-6, args.detuning_mhz * 1e6
    d, p, s = simulate_ramsey(t2_true, df_true, n_shots=shots, seed=seed + 1)
    r = fit_ramsey(d, p, s)
    print(f"Ramsey T2*    : {_fmt(r.T2, r.T2_err, 1e6, 'us')}   (injected {t2_true*1e6:.1f} us)")
    print(f"  detuning    : {_fmt(r.delta_f, r.delta_f_err, 1e-6, 'MHz')}   (injected {df_true/1e6:.3f} MHz)")

    # Rabi
    fr_true = args.rabi_mhz * 1e6
    d, p, s = simulate_rabi(fr_true, n_shots=shots, seed=seed + 2)
    r = fit_rabi(d, p, s)
    print(f"Rabi rate     : {_fmt(r.f_rabi, r.f_rabi_err, 1e-6, 'MHz')}   (injected {fr_true/1e6:.2f} MHz)")
    print(f"  pi pulse    : {_fmt(r.t_pi, r.t_pi_err, 1e9, 'ns')}")

    # Hahn echo vs Ramsey
    sigma_f, t2_int = args.noise_khz * 1e3, args.t2_intrinsic_us * 1e-6
    d, p, s = simulate_hahn_echo(sigma_f, t2_int, echo=False, n_shots=shots, seed=seed + 3)
    star = fit_hahn_echo(d, p, s)
    d, p, s = simulate_hahn_echo(sigma_f, t2_int, echo=True, n_shots=shots, seed=seed + 4)
    echo = fit_hahn_echo(d, p, s)
    print(f"Hahn echo     : T2*={star.T1*1e6:6.2f} us -> T2_echo={echo.T1*1e6:6.2f} us  (refocusing gain)")

    # Readout fidelity
    iq0, iq1 = simulate_readout_iq(args.snr, 1.0, n_shots=shots * 4, seed=seed + 5)
    rr = assignment_fidelity(iq0, iq1)
    print(f"Readout fid.  : F={rr.fidelity:.4f}  (F0={rr.f0:.4f}, F1={rr.f1:.4f}, SNR={rr.snr:.2f})")

    # Randomized benchmarking
    L, S, sg = simulate_rb(args.rb_p, n_sequences=30, n_shots=shots, seed=seed + 6)
    rb = fit_rb(L, S, sg)
    print(f"RB            : p={rb.p:.5f} +/- {rb.p_err:.5f}  EPC={rb.epc:.2e}  (injected p={args.rb_p})")

    print("=" * 60)
    print(f"QuTiP Lindblad engine available: {have_qutip()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-qubit characterization bench (T1/T2*/Rabi/echo/readout/RB)."
    )
    parser.add_argument("--shots", type=int, default=4096, help="single-shot reads per point")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t1-us", type=float, default=50.0)
    parser.add_argument("--t2-us", type=float, default=30.0)
    parser.add_argument("--detuning-mhz", type=float, default=0.5)
    parser.add_argument("--rabi-mhz", type=float, default=10.0)
    parser.add_argument("--noise-khz", type=float, default=80.0, help="quasi-static dephasing noise")
    parser.add_argument("--t2-intrinsic-us", type=float, default=100.0)
    parser.add_argument("--snr", type=float, default=6.0, help="readout blob separation / sigma")
    parser.add_argument("--rb-p", type=float, default=0.99, help="RB depolarizing parameter")
    args = parser.parse_args()
    run_battery(args)


if __name__ == "__main__":
    main()
