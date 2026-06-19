# Quantum Hardware Test Bench

A **single-qubit characterization bench** — the standard measurements used to
characterize and calibrate superconducting / trapped-ion qubits inside a dilution
refrigerator — together with the **cryostat thermal-control** instrumentation that
supports them.

Each experiment is simulated with realistic readout **shot noise** (binomial
sampling of N single-shot reads per point) and fit by **least squares**, reporting
every parameter with a **covariance-based uncertainty** (`perr = sqrt(diag(pcov))`)
— the same workflow run on real hardware.

## Characterization experiments

| Experiment | Model fit | Extracts |
|---|---|---|
| **T1** energy relaxation | `A·exp(-t/T1) + C` | T1 ± σ |
| **T2\*** Ramsey | `A·exp(-t/T2)·cos(2π·Δf·t + φ) + C` | T2\*, detuning Δf |
| **Rabi** calibration | `A/2·(1 − cos(2π·f_R·t)) + C` | Rabi frequency → π-pulse |
| **Hahn echo** T2 | refocused decay | T2_echo ( > T2\* ) |
| **Readout** assignment fidelity | IQ-plane Gaussian blobs | F, 2×2 confusion matrix |
| **Randomized benchmarking** | `A·p^m + B` | error-per-Clifford |

An optional **QuTiP Lindblad master-equation engine** lets T1/T2 emerge from
open-system dynamics (collapse operators `√Γ₁·σ⁻` and `√(Γφ/2)·σz`). It is imported
lazily — everything else runs on numpy/scipy alone, and the test suite is green
**without** QuTiP installed.

## Quickstart

```bash
pip install numpy scipy pandas matplotlib reportlab pytest

# Full characterization battery against simulated hardware (known params + noise):
python -m src.qubit --shots 8192
#   T1 relaxation : 49.7 +/- 0.6 us   (injected 50.0 us)
#   Ramsey T2*    : 29.8 +/- 0.5 us   ...
```

```python
from src.qubit import simulate_t1, fit_t1

delays, p_hat, sigma = simulate_t1(t1_true=50e-6, n_shots=4096, seed=0)
res = fit_t1(delays, p_hat, sigma)
print(res.T1, "+/-", res.T1_err)      # covariance-based 1σ error bar
```

## Cryostat thermal control (supporting infrastructure)

Qubits operate at ~4 K. The `cryocooler/` package is the cryostat controller that
holds that environment: PID temperature regulation against a lumped thermal model,
simulated sensors with noise/drift, a DAQ layer with fault recovery, and CSV/PDF
reporting. It is the environment the qubit bench runs in — supporting infrastructure,
not the headline.

## Package layout

```
src/
├── qubit/            # headline: T1 / T2* / Rabi / echo / readout / RB + fits with uncertainty
│   ├── models.py     #   shared fit models + covariance-based FitResult
│   ├── relaxation.py # ramsey.py  rabi.py  hahn_echo.py  readout.py
│   ├── randomized_benchmarking.py
│   └── lindblad.py   #   optional QuTiP master-equation engine
├── cryocooler/       # supporting cryostat thermal control (PID + thermal model + sensors)
├── daq/              # data acquisition + instrument comms
└── utils/            # reporting
```

## Testing

```bash
pytest tests/ -q     # every fit validated against injected ground truth
```

## License

MIT
