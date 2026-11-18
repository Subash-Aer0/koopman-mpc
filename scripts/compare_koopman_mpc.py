"""Koopman MPC vs physics-model MPC (both [LN-DDC] Eq. (17)), with the LQR for reference.

    PYTHONPATH=src python scripts/compare_koopman_mpc.py results/runs/<time>_outer_loop_koopman \
        configs/controllers/hover_mpc.toml configs/experiments/hover_gaussian.toml \
        configs/experiments/circle_slow_gaussian.toml configs/experiments/circle_7s_gaussian.toml \
        configs/experiments/circle_5s_gaussian.toml --seeds 5 --no-show \
        --ltv configs/controllers/ltv_mpc.toml

Controllers, all given the mean wind except the LQR:
    "MPC physics"   OutputMPC on the hover linearization (z = x − x*, C = I)
    "MPC Koopman"   OutputMPC on the identified lifted model
    "MPC full"      the baseline LinearMPC (Riccati terminal cost, model-consistent
                    attitude/input reference): what the physics model reaches with
                    the full design, for context
    "MPC LTV"       (with --ltv) LTVMPC: "MPC full" re-linearized along the
                    reference (same weights, limits and terminal mode)
    "LQR"           the existing state-dependent LQR (100 Hz, no wind knowledge)
The two Eq. (17) controllers share weights, horizon, input bounds, reference
(position and velocity only; attitude regulated to trim) and terminal weight Q,
so they differ only in (A, B, C, ψ). Every controller sees the same wind draws
for a given seed.

Writes summary.json (per experiment and controller: RMSE per seed, mean, std,
and the paired Koopman − physics difference) and figures per experiment plus a
speed-sweep figure for the circle experiments.
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.metrics import rmse
from analysis.mpc_plots import plot_sweep, plot_top_view, position_error
from analysis.plotting import plot_mse, plot_tracking_error, save_figures
from controllers.hover_model import HoverModel
from controllers.linear_mpc import outer_state
from controllers.output_mpc import OutputMPC, OutputMPCSettings, output_reference
from experiment.builders import load_experiment
from experiment.hover_mpc_config import load_hover_mpc, position_derivatives
from experiment.ltv_mpc_config import LTVMPCConfig, load_ltv_mpc
from experiment.run_record import start_run
from koopman.outer_model import OuterKoopmanModel, deviation_state
from simulation.closed_loop import simulate
from simulation.outer_loop import simulate_outer_loop

REPO_ROOT = Path(__file__).resolve().parents[1]
PHYSICS, KOOPMAN, FULL, LTV, LQR = "MPC physics", "MPC Koopman", "MPC full", "MPC LTV", "LQR"
logger = logging.getLogger("compare_koopman_mpc")


def run_controllers(experiment_path, seed, cfg, koopman: OuterKoopmanModel,
                    ltv: tuple[LTVMPCConfig, object] | None = None) -> dict:
    """One seed of one experiment: {label: SimulationLog}.

    ltv: (config, LTVModel built for this experiment), or None to skip LTV MPC.
    """
    logs = {}
    exp = load_experiment(experiment_path, seed=seed)
    logs[LQR] = simulate(exp.x0, exp.duration, exp.reference, exp.wind, exp.params, exp.lqr,
                         exp.rate_controller, exp.timing, actuator=exp.actuator)
    settings = OutputMPCSettings(Q=cfg.settings.Q, R=cfg.settings.R,
                                 horizon=cfg.settings.horizon, u_lb=cfg.settings.u_lb,
                                 u_ub=cfg.settings.u_ub)
    labels = (PHYSICS, KOOPMAN, FULL) + ((LTV,) if ltv is not None else ())
    for label in labels:
        run = load_experiment(experiment_path, seed=seed)   # fresh wind rng, same draws
        derivatives = position_derivatives(run.reference)
        if label == LTV:
            ltv_cfg, ltv_model = ltv
            controller = ltv_cfg.make_controller(ltv_model)   # fresh state, shared model
        elif label == FULL:
            controller = cfg.with_terminal("cost_only").make_controller(
                run.params, derivatives, run.duration)
        else:
            hover = HoverModel.build(run.params, cfg.wind_mean, cfg.yaw, cfg.sample_time)
            if not (np.allclose(koopman.x_trim, hover.x_trim)
                    and np.allclose(koopman.u_trim, hover.u_trim)):
                raise ValueError("the Koopman model was identified about another trim (e.g. the "
                                 "nominal one); compare it with scripts/compare_gap_closure.py")
            reference = output_reference(derivatives, hover.x_trim)
            if label == PHYSICS:
                lift = lambda x, h=hover: deviation_state(outer_state(x), None, h.x_trim, False)[0]
                controller = OutputMPC(hover.A, hover.B, np.eye(9), lift, hover.u_trim,
                                       settings, reference, hover.dt)
            else:
                controller = OutputMPC(koopman.A, koopman.B, koopman.C,
                                       koopman.lift_measurement, koopman.u_trim, settings,
                                       reference, koopman.dt)
        logs[label] = simulate_outer_loop(run.x0, run.duration, run.reference, run.wind,
                                          run.params, controller, run.rate_controller,
                                          dt_sim=run.timing.dt_sim, dt_log=run.timing.dt_log,
                                          actuator=run.actuator)
    return logs


def peak_tilt(experiment_path, gravity=9.81) -> float:
    """Peak tilt the reference needs, from its acceleration (0 for a setpoint)."""
    derivatives = position_derivatives(load_experiment(experiment_path).reference)
    t = np.linspace(0.0, 20.0, 401)
    a = np.array([derivatives(ti)[2] for ti in t])
    return float(np.max(np.arctan2(np.linalg.norm(a[:, :2], axis=1), gravity - a[:, 2])))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_run", type=Path, help="identify_outer_loop_model run directory")
    parser.add_argument("mpc", type=Path, help="hover MPC TOML (weights, horizon, limits, wind)")
    parser.add_argument("experiments", type=Path, nargs="+")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--ltv", type=Path, default=None,
                        help="LTV MPC TOML; adds 'MPC LTV' (its base config must be `mpc`)")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    parser.add_argument("--no-show", dest="show", action="store_false")
    args = parser.parse_args(argv)

    cfg = load_hover_mpc(args.mpc)
    koopman = OuterKoopmanModel.load(args.model_run)
    if not np.isclose(koopman.dt, cfg.sample_time) or \
            not np.allclose(koopman.wind_mean, cfg.wind_mean):
        raise ValueError("the Koopman model was identified at another sample time or mean wind")
    ltv_cfg = load_ltv_mpc(args.ltv) if args.ltv is not None else None
    if ltv_cfg is not None and ltv_cfg.base.config_path != cfg.config_path:
        raise ValueError(f"{args.ltv} is based on {ltv_cfg.base.config_path}, not {cfg.config_path}; "
                         "the MPCs would not share weights and limits")
    configs = [cfg.config_path] + (ltv_cfg.config_paths[:1] if ltv_cfg else [])
    run_dir = start_run(args.results_dir, "koopman_vs_physics_mpc", 0,
                        [*configs, *[p.resolve() for p in args.experiments]], REPO_ROOT)
    seeds = list(range(args.seeds))
    summary, figures, sweep = {"model_run": str(args.model_run.resolve()), "seeds": seeds}, {}, []

    for path in args.experiments:
        name = path.stem
        ltv = None
        if ltv_cfg is not None:   # the model does not depend on the seed: build it once
            exp = load_experiment(path)
            ltv = (ltv_cfg, ltv_cfg.make_model(exp.params, position_derivatives(exp.reference),
                                               exp.duration))
        per_seed = [run_controllers(path, seed, cfg, koopman, ltv) for seed in seeds]
        labels = list(per_seed[0])
        errors = {label: np.stack([position_error(logs[label]) for logs in per_seed])
                  for label in labels}
        rmse_3d = {label: np.linalg.norm(rmse(errors[label], axis=1), axis=1) for label in labels}
        paired = rmse_3d[KOOPMAN] - rmse_3d[PHYSICS]
        summary[name] = {
            "peak_reference_tilt_rad": peak_tilt(path),
            "rmse_3d_m": {label: {"per_seed": v.tolist(), "mean": float(v.mean()),
                                  "std": float(v.std())} for label, v in rmse_3d.items()},
            "rmse_per_axis_mean_m": {label: rmse(errors[label], axis=1).mean(axis=0).tolist()
                                     for label in labels},
            "koopman_minus_physics_m": {"per_seed": paired.tolist(), "mean": float(paired.mean()),
                                        "koopman_better_in": int(np.sum(paired < 0))},
        }
        if LTV in rmse_3d:
            vs_ltv = rmse_3d[KOOPMAN] - rmse_3d[LTV]
            summary[name]["koopman_minus_ltv_m"] = {
                "per_seed": vs_ltv.tolist(), "mean": float(vs_ltv.mean()),
                "koopman_better_in": int(np.sum(vs_ltv < 0))}
        logger.info("%s (peak tilt %.2f rad): %s | Koopman − physics %.4f m, better in %d/%d seeds",
                    name, summary[name]["peak_reference_tilt_rad"],
                    "  ".join(f"{label} {v.mean():.4f}±{v.std():.4f}" for label, v in rmse_3d.items()),
                    paired.mean(), int(np.sum(paired < 0)), len(seeds))
        t = per_seed[0][LQR].t
        figures[f"{name}_tracking_error"] = plot_tracking_error(t, errors, name)
        figures[f"{name}_mse"] = plot_mse(t, errors, baseline=PHYSICS, title=name)
        figures[f"{name}_top_view_seed0"] = plot_top_view(per_seed[0], f"{name} seed 0")
        if "circle" in name:
            sweep.append((summary[name]["peak_reference_tilt_rad"], rmse_3d))

    if len(sweep) > 1:
        sweep.sort(key=lambda item: item[0])
        tilts = [tilt for tilt, _ in sweep]
        results = {label: (np.array([r[label].mean() for _, r in sweep]),
                           np.array([r[label].std() for _, r in sweep])) for label in sweep[0][1]}
        figures["circle_speed_sweep"] = plot_sweep(
            tilts, "peak reference tilt [rad]", results,
            f"circle r = 2 m, mean ± std over {len(seeds)} wind seeds")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    save_figures(figures, run_dir / "figures")
    logger.info("results in %s", run_dir)
    if args.show:
        plt.show()
    plt.close("all")
    return run_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    main()
