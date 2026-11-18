"""Run the hover linear MPC (one or all terminal modes) and the LQR baseline on one experiment.

    PYTHONPATH=src python scripts/run_hover_mpc.py configs/experiments/circle_slow_gaussian.toml \
        configs/controllers/hover_mpc.toml configs/controllers/hover_mpc_no_wind.toml \
        --terminal all --no-show

Several MPC configs may be given (e.g. with and without mean-wind knowledge);
--horizon overrides N in all of them (horizon ablations).

Every controller sees the same wind draws (same seed). Writes the run record of
experiment.run_record plus, per controller, <label>/log.npz and for MPC
<label>/mpc_log.npz, and summary.json with per-axis RMSE, fallbacks and solve
times. Figures go to <run_dir>/figures/.
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.metrics import rmse
from analysis.mpc_plots import plot_commands, plot_solver, plot_top_view, position_error
from analysis.plotting import plot_tracking, plot_tracking_error, save_figures
from controllers.linear_mpc import TERMINAL_MODES
from dynamics import dynamics as rigid_body
from dynamics.quaternion import Quaternion
from experiment.builders import load_experiment
from experiment.hover_mpc_config import load_hover_mpc, position_derivatives
from experiment.run_record import save_log, start_run
from simulation.closed_loop import simulate
from simulation.outer_loop import simulate_outer_loop

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("run_hover_mpc")


def _summary(log, mpc_log=None) -> dict:
    axis_rmse = rmse(position_error(log), axis=0)
    out = {"rmse_m": axis_rmse.tolist(), "rmse_3d_m": float(np.linalg.norm(axis_rmse))}
    if mpc_log is not None:
        out.update(fallbacks=int(mpc_log["fallback"].sum()), solves=int(len(mpc_log["t"])),
                   solve_time_mean_ms=float(1e3 * mpc_log["solve_time"].mean()),
                   solve_time_max_ms=float(1e3 * mpc_log["solve_time"].max()))
    return out


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", type=Path, help="experiment TOML file")
    parser.add_argument("mpc", type=Path, nargs="+", help="hover MPC TOML file(s)")
    parser.add_argument("--terminal", default=None,
                        help=f"one of {TERMINAL_MODES}, or 'all'; default: the config's")
    parser.add_argument("--horizon", type=int, default=None, help="override N in every config")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    parser.add_argument("--no-show", dest="show", action="store_false")
    args = parser.parse_args(argv)

    exp = load_experiment(args.experiment)
    cfgs = [load_hover_mpc(path) for path in args.mpc]
    if args.horizon is not None:
        cfgs = [cfg.with_horizon(args.horizon) for cfg in cfgs]
    reference_yaw = Quaternion(exp.reference(0.0)[0][3:7]).to_euler()[2]
    for cfg in cfgs:
        if not np.isclose(reference_yaw, cfg.yaw):
            raise ValueError(f"reference yaw {reference_yaw} rad differs from the MPC trim yaw "
                             f"{cfg.yaw} rad in {cfg.config_path.name}; the hover model is "
                             "linearized at fixed yaw")

    run_dir = start_run(args.results_dir, f"{exp.name}_hover_mpc", exp.seed,
                        exp.config_paths + [cfg.config_path for cfg in cfgs], REPO_ROOT)
    logs, mpc_logs, summary = {}, {}, {}

    lqr_exp = load_experiment(args.experiment)
    logs["LQR"] = simulate(lqr_exp.x0, lqr_exp.duration, lqr_exp.reference, lqr_exp.wind,
                           lqr_exp.params, lqr_exp.lqr, lqr_exp.rate_controller, lqr_exp.timing,
                           actuator=lqr_exp.actuator)
    summary["LQR"] = _summary(logs["LQR"])

    runs = [(cfg, mode) for cfg in cfgs for mode in
            (TERMINAL_MODES if args.terminal == "all" else [args.terminal or cfg.settings.terminal])]
    for cfg, mode in runs:
        run = load_experiment(args.experiment)   # fresh wind rng: same draws for everyone
        mpc = cfg.with_terminal(mode).make_controller(
            run.params, position_derivatives(run.reference), run.duration)
        label = f"MPC {mode}" + (f" {cfg.config_path.stem}" if len(cfgs) > 1 else "")
        logs[label] = simulate_outer_loop(run.x0, run.duration, run.reference, run.wind,
                                          run.params, mpc, run.rate_controller,
                                          dt_sim=run.timing.dt_sim, dt_log=run.timing.dt_log,
                                          actuator=run.actuator)
        mpc_logs[label] = mpc.log.arrays()
        summary[label] = _summary(logs[label], mpc_logs[label])
        if mpc.terminal_set is not None:
            summary[label]["terminal_set_rows"] = int(len(mpc.terminal_set[1]))
        summary[label]["trim_euler_rad"] = mpc.model.x_trim[6:9].tolist()
        summary[label]["trim_thrust"] = float(mpc.model.u_trim[3])
        summary[label]["horizon"] = cfg.settings.horizon

    for label, log in logs.items():
        directory = run_dir / label.replace(" ", "_")
        directory.mkdir()
        save_log(directory, log)
        if label in mpc_logs:
            np.savez(directory / "mpc_log.npz", **mpc_logs[label])
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    for label, s in summary.items():
        extra = f", {s['fallbacks']}/{s['solves']} fallbacks, solve mean " \
                f"{s['solve_time_mean_ms']:.2f} ms max {s['solve_time_max_ms']:.2f} ms" \
            if "fallbacks" in s else ""
        logger.info("%-16s 3-D RMSE %.4f m  per axis %s%s", label, s["rmse_3d_m"],
                    np.round(s["rmse_m"], 4).tolist(), extra)

    t = logs["LQR"].t
    figures = {
        "tracking_error": plot_tracking_error(
            t, {label: position_error(log)[None] for label, log in logs.items()}, exp.name),
        "top_view": plot_top_view(logs, exp.name),
        "commands": plot_commands(logs, cfgs[0].settings.u_lb, cfgs[0].settings.u_ub, exp.name),
        "solver": plot_solver(mpc_logs, cfgs[0].sample_time, exp.name),
    }
    for label, log in logs.items():
        for key, fig in plot_tracking(log, title=f"{exp.name} {label}").items():
            figures[f"{key}_{label.replace(' ', '_')}"] = fig
    paths = save_figures(figures, run_dir / "figures")
    logger.info("run %s, %d figures", run_dir, len(paths))
    if args.show:
        plt.show()
    plt.close("all")
    return run_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    main()
