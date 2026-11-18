"""Re-plot saved runs from their log.npz into <run_dir>/figures/, then show them.

    PYTHONPATH=src python scripts/plot_run.py results/runs/<run_dir> [<run_dir> ...]

Pass --no-show to only save the figures.
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from analysis.plotting import plot_tracking, save_figures
from experiment.run_record import load_log

logger = logging.getLogger("plot_run")


def main(argv: list[str] | None = None) -> list[Path]:
    """Plot each run directory; returns the written figure paths."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", type=Path, nargs="+", help="run directories with log.npz")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="save figures without opening windows")
    args = parser.parse_args(argv)

    written = []
    for run_dir in args.run_dirs:
        paths = save_figures(plot_tracking(load_log(run_dir), title=run_dir.name),
                             run_dir / "figures")
        logger.info("%s: %s", run_dir.name, ", ".join(path.name for path in paths))
        written.extend(paths)
    if args.show:
        plt.show()
    plt.close("all")
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    main()
