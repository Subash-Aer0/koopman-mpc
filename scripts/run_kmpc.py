"""Run LQR + K-MPC augmentation on one experiment config, save a run record and plot it.

    PYTHONPATH=src python scripts/run_kmpc.py configs/experiments/hover_wind.toml \
        configs/controllers/kmpc.toml

Writes the run record of experiment.run_record plus:
    mpc_log.npz                MPCLog: every solve, its status and plan
    kmpc_identification.json   copy of the model run's identification report
Figures are saved to <run_dir>/figures/ and then shown on screen; pass
--no-show for headless runs.
"""
import argparse
import logging
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np

from analysis.plotting import plot_tracking, save_figures
from controllers.sdlqr_model import POS as REF_POS
from dynamics.dynamics import POS
from experiment.augmentation import KMPCAugmentation
from experiment.builders import load_experiment, load_kmpc
from experiment.run_record import save_log, start_run
from simulation.closed_loop import simulate

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("run_kmpc")


def main(argv: list[str] | None = None) -> Path:
    """Simulate the experiment with K-MPC, write results/runs/<time>_<name>_kmpc/, return it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", type=Path, help="experiment TOML file")
    parser.add_argument("kmpc", type=Path, help="K-MPC TOML file")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="save figures without opening windows")
    args = parser.parse_args(argv)

    exp = load_experiment(args.experiment)
    kmpc = load_kmpc(args.kmpc)
    kmpc.check_timing(exp.timing)
    name = f"{exp.name}_kmpc"
    run_dir = start_run(args.results_dir, name, exp.seed,
                        exp.config_paths + kmpc.config_paths, REPO_ROOT)
    shutil.copyfile(kmpc.model_run / "identification.json", run_dir / "kmpc_identification.json")
    logger.info("run %s: seed %d, %.1f s simulated, model %s",
                run_dir.name, exp.seed, exp.duration, kmpc.model_run.name)

    augmentation = KMPCAugmentation(kmpc.make_controller())
    log = simulate(exp.x0, exp.duration, exp.reference, exp.wind, exp.params,
                   exp.lqr, exp.rate_controller, exp.timing, augmentation=augmentation,
                   actuator=exp.actuator)
    save_log(run_dir, log)
    mpc_log = augmentation.log()
    mpc_log.save(run_dir / "mpc_log.npz")

    final_error = np.linalg.norm(log.x[-1, POS] - log.x_ref[-1, REF_POS])
    logger.info("final position error %.4f m; %d of %d solves fell back to LQR; "
                "K-MPC step time mean %.2f ms, max %.2f ms (budget %.0f ms)",
                final_error, augmentation.fallback_count, len(mpc_log.t),
                1e3 * mpc_log.solve_time.mean(), 1e3 * mpc_log.solve_time.max(),
                1e3 * exp.timing.dt_augment)
    if augmentation.fallback_count:
        logger.warning("K-MPC fell back to LQR at t = %s s",
                       np.round(mpc_log.t[~mpc_log.solved], 2).tolist())

    figure_paths = save_figures(plot_tracking(log, title=name), run_dir / "figures")
    logger.info("figures: %s", ", ".join(path.name for path in figure_paths))
    if args.show:
        plt.show()
    plt.close("all")
    return run_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    main()
