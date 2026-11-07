"""Run the baseline LQR on one experiment config, save a run record and plot it.

    PYTHONPATH=src python scripts/run_baseline.py configs/experiments/figure8_wind.toml

Figures are saved to <run_dir>/figures/ and then shown on screen (the script
blocks until the windows are closed); pass --no-show for headless runs.
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plotting import plot_tracking, save_figures
from controllers.sdlqr_model import POS as REF_POS
from dynamics.dynamics import POS
from experiment.builders import load_experiment
from experiment.run_record import save_log, start_run
from simulation.closed_loop import simulate

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("run_baseline")


def main(argv: list[str] | None = None) -> Path:
    """Simulate the experiment, write results/runs/<time>_<name>/, return that directory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", type=Path, help="experiment TOML file")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="save figures without opening windows")
    args = parser.parse_args(argv)

    exp = load_experiment(args.experiment)
    run_dir = start_run(args.results_dir, exp.name, exp.seed, exp.config_paths, REPO_ROOT)
    logger.info("run %s: seed %d, %.1f s simulated", run_dir.name, exp.seed, exp.duration)

    log = simulate(exp.x0, exp.duration, exp.reference, exp.wind, exp.params,
                   exp.lqr, exp.rate_controller, exp.timing)
    save_log(run_dir, log)

    final_error = np.linalg.norm(log.x[-1, POS] - log.x_ref[-1, REF_POS])
    logger.info("final position error %.4f m; saved to %s", final_error, run_dir)

    figure_paths = save_figures(plot_tracking(log, title=exp.name), run_dir / "figures")
    logger.info("figures: %s", ", ".join(path.name for path in figure_paths))
    if args.show:
        plt.show()
    plt.close("all")
    return run_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    main()
