"""Fit the grey-box body drag coefficients to an outer-loop dataset.

    PYTHONPATH=src python scripts/fit_greybox_drag.py configs/identify/greybox_drag.toml \
        results/runs/<time>_outer_loop_data

Fits on the train split, reports the residual on both splits, and writes
results/runs/<time>_greybox_drag/greybox_drag.json with the coefficients.
The simulator's own coefficients are reported for reference only; the fit
never sees them. Load the result with identification.greybox_drag.load_greybox_drag.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

from datagen.outer_loop import load_episodes
from dynamics.params import load_params
from experiment.builders import load_toml
from experiment.ltv_mpc_config import load_ltv_mpc
from experiment.run_record import start_run
from identification.greybox_drag import (
    drag_regression, fit_drag, residual_rms, save_greybox_drag,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("fit_greybox_drag")


def stacked_regression(episodes, dt: float, params):
    parts = [drag_regression(ep.x9, ep.u, ep.v_w, dt, params.mass, params.gravity)
             for ep in episodes]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("dataset", type=Path, help="collect_outer_loop_data run directory")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    args = parser.parse_args(argv)

    path = args.config.resolve()
    section = load_toml(path)["greybox"]
    mpc_cfg = load_ltv_mpc(path.parent / section["mpc_config"])
    dataset_dir = args.dataset.resolve()
    meta = json.loads((dataset_dir / "metadata.json").read_text())
    vehicle_path = next(Path(p) for p in meta["configs"] if Path(p).name == "quadrotor.toml")
    params = load_params(vehicle_path)   # mass and gravity are known; drag is not used
    dt = mpc_cfg.base.sample_time
    run_dir = start_run(args.results_dir, section["name"], int(section["seed"]),
                        [path, *mpc_cfg.config_paths], REPO_ROOT)

    splits = load_episodes(dataset_dir)
    train = stacked_regression(splits["train"], dt, params)
    valid = stacked_regression(splits["validation"], dt, params)
    drag = fit_drag(*train, str(section["form"]))
    result = {
        "dataset": str(dataset_dir),
        "form": section["form"],
        "drag_coeffs": drag.tolist(),
        "residual_rms_m_s2": {"train": residual_rms(*train, drag).tolist(),
                              "validation": residual_rms(*valid, drag).tolist()},
        "simulator_drag_coeffs_for_reference": params.drag_coeffs.tolist(),
    }
    save_greybox_drag(run_dir, result)
    logger.info("drag %s N·s²/m² (simulator %s); validation residual %s m/s²",
                np.round(drag, 4), params.drag_coeffs, np.round(result["residual_rms_m_s2"]
                                                                ["validation"], 4))
    logger.info("result in %s", run_dir)
    return run_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    main()
