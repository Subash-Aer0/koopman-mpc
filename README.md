# koopman-mpc

Reproduction of *Data-Driven Koopman-based Modeling and Control of Quadcopter in a
Turbulent Wind Field* (Tabassum & Bai, IEEE CCTA 2023).

## Setup

```bash
conda env create -f environment.yml
conda activate koopman_mpc
```

Scripts and tests expect `src/` on the path, e.g. `PYTHONPATH=src python scripts/01_generate_data.py`.

## Layout

| Path | Paper | Contents |
|---|---|---|
| `src/config.py` | Sec. IV | Values reported in the paper (dt, v̄w, Σvw, λ, N, np, β) |
| `src/dynamics/` | Sec. II-A, Eqs. (1)-(4) | 13-state quadcopter model, quaternions, quadratic drag, quadratic rotor model + hybrid BEMT |
| `src/wind/` | Sec. II-A, IV-A | Mean + Gaussian wind, LES wind replay |
| `src/controllers/lqr.py`, `low_level.py` | Sec. III-A | Baseline high-level LQR and low-level rate controller |
| `src/koopman/` | Sec. III-B, Eqs. (5)-(11) | Monomial lifting, l1-regularized EDMD, Model 1 / Model 2 |
| `src/datagen/` | Fig. 1, Sec. IV-A | Reference trajectories, perturbed closed-loop data collection, residual thrust |
| `src/controllers/kmpc.py` | Eqs. (15)-(16) | K-MPC (mean wind) |
| `src/controllers/ksmpc.py` | Eqs. (17)-(19) | K-SMPC (chance constraints, tightening γ) |
| `src/controllers/qp.py` | Sec. III-C | Dense QP condensing for both MPCs |
| `src/simulation/` | Sec. IV-C | Closed-loop simulation and Monte Carlo runs |
| `src/analysis/` | Tables I-III, Figs. 2-4 | Metrics and plots |
| `scripts/` | | Pipeline: generate data → identify models → evaluate controllers |
| `data/wind/` | | LES wind fields (input) |
| `data/raw/`, `data/processed/` | | Generated datasets (git-ignored) |
| `results/` | | Saved models, figures, tables (git-ignored) |
| `tests/` | | Unit tests |

## Pipeline

1. `scripts/01_generate_data.py`: 2 reference trajectories × 4 perturbation/wind repeats for training, plus 2 validation sets, all at 10 Hz.
2. `scripts/02_identify_models.py`: fit Models 1 and 2 (N = 55, p = 2, λ = 20). Produces Table I and Fig. 2.
3. `scripts/03_evaluate_controllers.py`: LQR vs LQR+K-MPC vs LQR+K-SMPC over 5 Monte Carlo runs. Produces Tables II-III and Figs. 3-4.

## Not specified in the paper

Vehicle mass and inertia, rotor and drag coefficients, LQR/MPC weights Q and R, input and state bounds, perturbation magnitudes, the reference trajectories, and the LES wind data [26]. These are marked `TODO` in the code.
