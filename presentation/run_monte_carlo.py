"""Monte Carlo runs of the full-MPC comparison that keep every time series (for the slides).

    PYTHONPATH=src python presentation/run_monte_carlo.py \
        results/runs/<time>_outer_loop_koopman results/runs/<time>_greybox_drag \
        configs/experiments/circle_7s_gaussian.toml configs/experiments/figure8_10s_gaussian.toml \
        --seeds 10

Same controllers, wind draws and settings as scripts/compare_gap_closure.py (its
run_controllers is reused, so the two cannot drift apart). compare_gap_closure
keeps only position errors; this keeps the full state, reference, command and
wind of every run, so the slides can show attitude and velocity errors and the
commands themselves.

Writes, per experiment, <name>.npz in a run directory:
    labels (C,), seeds (S,), t (T,)               s
    x      (S, C, T, 13)  rigid-body state, layout in dynamics.py (NED, quaternion w-first, FRD)
    x_ref  (S, C, T, 10)  reference [p, q, v], layout in sdlqr_model.py
    u_cmd  (S, C, T, 4)   [ω (rad/s, body FRD), c (m/s²)] sent to the rate loop
    v_w    (S, C, T, 3)   wind, world NED, m/s
    u_lb, u_ub (4,)       MPC command bounds
"""
import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compare_gap_closure as gap  # noqa: E402  (scripts/ is not a package)
from experiment.builders import load_experiment  # noqa: E402
from experiment.hover_mpc_config import position_derivatives  # noqa: E402
from experiment.ltv_mpc_config import load_ltv_mpc  # noqa: E402
from experiment.run_record import start_run  # noqa: E402
from identification.greybox_drag import load_greybox_drag  # noqa: E402
from koopman.outer_model import OuterKoopmanModel  # noqa: E402

NOMINAL = REPO_ROOT / "configs/controllers/ltv_mpc_nominal.toml"
TRUE_DRAG = REPO_ROOT / "configs/controllers/ltv_mpc.toml"
logger = logging.getLogger("run_monte_carlo")


def _one_seed(args: tuple) -> dict:
    """Worker: all controllers on one seed of one experiment, as plain arrays."""
    experiment_path, seed, model_run, greybox_run = args
    koopman = OuterKoopmanModel.load(model_run)
    nominal_cfg, true_drag_cfg = load_ltv_mpc(NOMINAL), load_ltv_mpc(TRUE_DRAG)
    greybox_cfg = nominal_cfg.with_drag_coeffs(load_greybox_drag(greybox_run))
    exp = load_experiment(experiment_path)
    derivatives = position_derivatives(exp.reference)
    cfgs = {gap.LTV_NOMINAL: nominal_cfg, gap.LTV_TRUE_DRAG: true_drag_cfg,
            gap.LTV_GREYBOX: greybox_cfg}
    ltvs = {label: (cfg, cfg.make_model(exp.params, derivatives, exp.duration))
            for label, cfg in cfgs.items()}
    logs = gap.run_controllers(experiment_path, seed, koopman, ltvs)
    return {label: {k: getattr(log, k) for k in ("t", "x", "x_ref", "u_cmd", "v_w")}
            for label, log in logs.items()}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_run", type=Path, help="identify_outer_loop_model run directory")
    parser.add_argument("greybox_run", type=Path, help="fit_greybox_drag run directory")
    parser.add_argument("experiments", type=Path, nargs="+")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    args = parser.parse_args(argv)

    experiments = [p.resolve() for p in args.experiments]
    gap.check_compatible(OuterKoopmanModel.load(args.model_run), load_ltv_mpc(NOMINAL),
                         load_ltv_mpc(TRUE_DRAG), load_experiment(experiments[0]).params)
    run_dir = start_run(args.results_dir, "presentation_mc", 0,
                        [NOMINAL, TRUE_DRAG, *experiments], REPO_ROOT)
    seeds = list(range(args.seeds))
    base = load_ltv_mpc(NOMINAL).base
    (run_dir / "inputs.json").write_text(json.dumps(
        {"model_run": str(args.model_run.resolve()),
         "greybox_run": str(args.greybox_run.resolve()), "seeds": seeds}, indent=2))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for path in experiments:
            jobs = [(path, s, args.model_run, args.greybox_run) for s in seeds]
            per_seed = list(pool.map(_one_seed, jobs))
            labels = list(per_seed[0])
            stacked = {key: np.stack([np.stack([run[label][key] for label in labels])
                                      for run in per_seed])
                       for key in ("x", "x_ref", "u_cmd", "v_w")}
            np.savez(run_dir / f"{path.stem}.npz", labels=np.array(labels),
                     seeds=np.array(seeds), t=per_seed[0][labels[0]]["t"],
                     u_lb=base.settings.u_lb, u_ub=base.settings.u_ub, **stacked)
            logger.info("%s: %d seeds x %d controllers saved", path.stem, len(seeds), len(labels))
    logger.info("results in %s", run_dir)
    return run_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    main()
