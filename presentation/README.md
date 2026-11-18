# Presentation: Koopman MPC for a quadrotor in turbulent wind

| File | Role |
|---|---|
| `theme.typ` | Self-contained academic slide theme (no Typst packages, compiles offline) |
| `slides.typ` | The deck; every table number is read from `data/*.json` |
| `run_monte_carlo.py` | Re-runs the full-MPC comparison (same code path as `scripts/compare_gap_closure.py`) and keeps full time series |
| `make_figures.py` | Reads saved runs only; writes `figures/*.svg` and `data/*.json` |

Build (from the repo root):

```bash
conda activate koopman_mpc
PYTHONPATH=src python presentation/run_monte_carlo.py \
    results/runs/20261118T194120Z_outer_loop_koopman results/runs/20261118T194118Z_greybox_drag \
    configs/experiments/circle_7s_gaussian.toml configs/experiments/figure8_10s_gaussian.toml --seeds 10
PYTHONPATH=src python presentation/make_figures.py \
    --paper-run results/runs/20261113T161039Z_kmpc_vs_lqr \
    --mc-run results/runs/<time>_presentation_mc \
    --gap-run results/runs/20261118T194126Z_gap_closure \
    --model-run results/runs/20261118T194120Z_outer_loop_koopman \
    --greybox-run results/runs/20261118T194118Z_greybox_drag \
    --dataset-run results/runs/20261118T192413Z_outer_loop_data \
    --bemt-run results/runs/20261118T184125Z_rotor_bemt
typst compile presentation/slides.typ
```

Caveat: the paper-reproduction run (`kmpc_vs_lqr`) predates the BEMT rotors, so it
flew with ideal rotors; the full-MPC runs use BEMT rotors.
