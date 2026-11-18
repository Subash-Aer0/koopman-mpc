"""Plot Figs. 3-4 of a saved controller evaluation into <run_dir>/figures/, then show them.

    PYTHONPATH=src python scripts/plot_evaluation.py results/runs/<time>_kmpc_vs_lqr

Reads position_errors.npz (keys "<experiment>/<controller>", (runs, T, 3)) and
takes t from the experiment's first logged episode; writes
<experiment>_abs_error.png (Fig. 3) and <experiment>_mse.png (Fig. 4).
Pass --no-show to only save the figures.
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plotting import plot_mse, plot_tracking_error, save_figures
from experiment.run_record import load_log

logger = logging.getLogger("plot_evaluation")
BASELINE = "lqr"
LABELS = {"lqr": "LQR", "kmpc": "LQR + K-MPC"}


def load_errors(run_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """experiment -> controller -> e_p (runs, T, 3), in the order they were saved."""
    grouped: dict[str, dict[str, np.ndarray]] = {}
    with np.load(run_dir / "position_errors.npz") as data:
        for key in data.files:
            experiment, controller = key.split("/")
            grouped.setdefault(experiment, {})[controller] = data[key]
    if not grouped:
        raise ValueError(f"{run_dir / 'position_errors.npz'} has no errors")
    return grouped


def main(argv: list[str] | None = None) -> list[Path]:
    """Plot every experiment of the evaluation run; returns the written figure paths."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="evaluation run directory")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="save figures without opening windows")
    args = parser.parse_args(argv)

    written = []
    for experiment, by_controller in load_errors(args.run_dir).items():
        if BASELINE not in by_controller:
            raise ValueError(f"{experiment}: no '{BASELINE}' baseline to compare against")
        first = next(iter(by_controller))
        t = load_log(args.run_dir / "episodes" / experiment / first / "run_00").t
        labelled = {LABELS.get(name, name): e for name, e in by_controller.items()}
        figures = {
            f"{experiment}_abs_error": plot_tracking_error(t, labelled, title=experiment),
            f"{experiment}_mse": plot_mse(t, labelled, baseline=LABELS[BASELINE],
                                          title=experiment),
        }
        paths = save_figures(figures, args.run_dir / "figures")
        logger.info("%s: %s", experiment, ", ".join(path.name for path in paths))
        written.extend(paths)
    if args.show:
        plt.show()
    plt.close("all")
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    main()
