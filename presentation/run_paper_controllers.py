"""LQR and LQR + K-Res-MPC (the paper's architecture) on the full-MPC references.

    PYTHONPATH=src python presentation/run_paper_controllers.py configs/controllers/kmpc_bemt.toml \
        configs/experiments/hover_gaussian.toml ... --seeds 10

Seeds are 0 .. S−1 and each run builds its episode with load_experiment(path, seed),
exactly as presentation/run_monte_carlo.py and scripts/compare_gap_closure.py do,
so for one seed all controllers of all three scripts fly the same wind. Controllers
are built as in scripts/03_evaluate_controllers.py (simulation.monte_carlo.run_paired,
experiment.augmentation.KMPCAugmentation). The vehicle's rotor model (ideal or BEMT)
comes from the experiment's vehicle config.

Writes, per experiment, <name>.npz in a run directory, in run_monte_carlo.py's layout:
    labels (C,) = ["LQR", "K-Res-MPC"], seeds (S,), t (T,) s
    x (S, C, T, 13), x_ref (S, C, T, 10), u_cmd (S, C, T, 4), v_w (S, C, T, 3)
u_cmd is the command sent to the rate loop, sat(u_h + u_d).
"""
import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiment.augmentation import KMPCAugmentation
from experiment.builders import load_experiment, load_kmpc
from experiment.run_record import start_run
from simulation.monte_carlo import run_paired

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = {"lqr": "LQR", "kmpc": "K-Res-MPC"}
logger = logging.getLogger("run_paper_controllers")


def _one_seed(args: tuple) -> dict:
    """Worker: LQR and LQR + K-Res-MPC on one seed of one experiment, as plain arrays."""
    experiment_path, seed, kmpc_path = args
    kmpc = load_kmpc(kmpc_path)
    controllers = {"lqr": lambda: None,
                   "kmpc": lambda: KMPCAugmentation(kmpc.make_controller())}
    results = run_paired(lambda s: load_experiment(experiment_path, s), controllers, [seed])
    return {LABELS[name]: {k: getattr(runs[0].log, k) for k in ("t", "x", "x_ref", "u_cmd", "v_w")}
            for name, runs in results.items()}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kmpc", type=Path, help="K-MPC config (configs/controllers/kmpc*.toml)")
    parser.add_argument("experiments", type=Path, nargs="+")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    args = parser.parse_args(argv)

    kmpc = load_kmpc(args.kmpc)
    experiments = [p.resolve() for p in args.experiments]
    config_paths = list(kmpc.config_paths)
    for path in experiments:
        exp = load_experiment(path)
        kmpc.check_timing(exp.timing)
        config_paths += exp.config_paths
    run_dir = start_run(args.results_dir, "paper_controllers", 0,
                        list(dict.fromkeys(config_paths)), REPO_ROOT)
    seeds = list(range(args.seeds))
    (run_dir / "inputs.json").write_text(json.dumps(
        {"kmpc_config": str(args.kmpc.resolve()), "model_run": str(kmpc.model_run),
         "actuator": load_experiment(experiments[0]).actuator.model, "seeds": seeds}, indent=2))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for path in experiments:
            per_seed = list(pool.map(_one_seed, [(path, s, args.kmpc.resolve()) for s in seeds]))
            labels = list(per_seed[0])
            stacked = {key: np.stack([np.stack([run[label][key] for label in labels])
                                      for run in per_seed])
                       for key in ("x", "x_ref", "u_cmd", "v_w")}
            np.savez(run_dir / f"{path.stem}.npz", labels=np.array(labels), seeds=np.array(seeds),
                     t=per_seed[0][labels[0]]["t"], **stacked)
            logger.info("%s: %d seeds x %d controllers saved", path.stem, len(seeds), len(labels))
    logger.info("results in %s", run_dir)
    return run_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    main()
