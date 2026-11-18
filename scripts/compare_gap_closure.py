"""How much better than the nominal controller do the learned models fly?

    PYTHONPATH=src python scripts/compare_gap_closure.py results/runs/<time>_outer_loop_koopman \
        configs/controllers/ltv_mpc_nominal.toml configs/controllers/ltv_mpc.toml \
        configs/experiments/circle_slow_gaussian.toml configs/experiments/circle_7s_gaussian.toml \
        configs/experiments/circle_5s_gaussian.toml --seeds 5 --no-show \
        --greybox results/runs/<time>_greybox_drag

Controllers, all at the MPC sample time, on the same wind draws for a given seed:
    "LTV nominal"     LTVMPC whose model has no drag: the controller available
                      before learning (lower bound)
    "Output nominal"  OutputMPC ([LN-DDC] Eq. 17) on the nominal hover model: the
                      same controller as "Koopman", so the two differ only in the model
    "Koopman"         OutputMPC on the identified lifted model
    "LTV grey-box"    (with --greybox) LTVMPC believing drag coefficients fitted
                      from flight data (scripts/fit_greybox_drag.py)
    "LTV true drag"   LTVMPC with the simulator's own body-drag model (simulation
                      only). With ideal rotors this is the oracle; with BEMT rotors
                      (vehicle config [rotor] model = "bemt") it misses the rotor
                      effects and is no longer an upper bound.

    reduction  = 100 (RMSE_LTV nominal − RMSE_X) / RMSE_LTV nominal, %, every X
    gap closed = (RMSE_LTV nominal − RMSE_X) / (RMSE_LTV nominal − RMSE_LTV true drag),
                 X = Koopman and the grey-box; > 1 means better than the true-drag LTV

Caveat: the LTVs run the full design (Riccati terminal cost, model-consistent
attitude and input reference) while Koopman runs Eq. (17) (terminal weight Q,
attitude regulated to trim). "Output nominal" vs "Koopman" is the like-for-like
model comparison; the gap fraction also contains this controller difference.

The Koopman model must be identified in the nominal coordinates (deviations
from the nominal hover trim, identify config mpc_config = ltv_mpc_nominal.toml).

Writes summary.json (per experiment and controller: RMSE per seed, mean, std,
per-axis RMSE, % reduction from nominal, and the gap closed per seed and of the
means) and figures. The actuator of the vehicle config (ideal or BEMT) is recorded.
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.metrics import gap_closed, percent_reduction, rmse
from analysis.mpc_plots import plot_top_view, position_error
from analysis.plotting import plot_mse, plot_tracking_error, save_figures
from controllers.linear_mpc import outer_state
from controllers.output_mpc import OutputMPC, OutputMPCSettings, output_reference
from experiment.builders import load_experiment
from experiment.hover_mpc_config import position_derivatives
from experiment.ltv_mpc_config import LTVMPCConfig, load_ltv_mpc
from experiment.run_record import start_run
from identification.greybox_drag import load_greybox_drag
from koopman.outer_model import OuterKoopmanModel, deviation_state
from simulation.outer_loop import simulate_outer_loop

REPO_ROOT = Path(__file__).resolve().parents[1]
LTV_NOMINAL, OUTPUT_NOMINAL, KOOPMAN, LTV_GREYBOX, LTV_TRUE_DRAG = (
    "LTV nominal", "Output nominal", "Koopman", "LTV grey-box", "LTV true drag")
logger = logging.getLogger("compare_gap_closure")


def check_compatible(koopman: OuterKoopmanModel, nominal: LTVMPCConfig, true_drag: LTVMPCConfig,
                     params) -> None:
    """Raise unless the three controllers share base settings and the Koopman coordinates."""
    if nominal.base.config_path != true_drag.base.config_path:
        raise ValueError(f"nominal and true-drag LTVs use different base configs "
                         f"({nominal.base.config_path}, {true_drag.base.config_path}); "
                         "they would not share weights and limits")
    if not nominal.model_drag_scale < true_drag.model_drag_scale:
        raise ValueError(f"the nominal model's drag scale {nominal.model_drag_scale} is not "
                         f"below the true-drag LTV's {true_drag.model_drag_scale}")
    base = nominal.base
    if not (np.isclose(koopman.dt, base.sample_time)
            and np.allclose(koopman.wind_mean, base.wind_mean)):
        raise ValueError("the Koopman model was identified at another sample time or mean wind")
    hover = nominal.hover_model(params)
    if not (np.allclose(koopman.x_trim, hover.x_trim)
            and np.allclose(koopman.u_trim, hover.u_trim)):
        raise ValueError("the Koopman model was not identified about the nominal hover trim")


def run_controllers(experiment_path: Path, seed: int, koopman: OuterKoopmanModel,
                    ltvs: dict[str, tuple]) -> dict:
    """One seed of one experiment: {label: SimulationLog}.

    ltvs: label -> (LTVMPCConfig, LTVModel built for this experiment); must
        contain LTV_NOMINAL and LTV_TRUE_DRAG, may contain LTV_GREYBOX.
    """
    nominal_cfg = ltvs[LTV_NOMINAL][0]
    base = nominal_cfg.base
    settings = OutputMPCSettings(Q=base.settings.Q, R=base.settings.R,
                                 horizon=base.settings.horizon, u_lb=base.settings.u_lb,
                                 u_ub=base.settings.u_ub)
    logs = {}
    order = (LTV_NOMINAL, OUTPUT_NOMINAL, KOOPMAN, LTV_GREYBOX, LTV_TRUE_DRAG)
    labels = [label for label in order if label in ltvs or label in (OUTPUT_NOMINAL, KOOPMAN)]
    for label in labels:
        run = load_experiment(experiment_path, seed=seed)   # fresh wind rng, same draws
        if label in ltvs:
            cfg, model = ltvs[label]
            controller = cfg.make_controller(model)
        else:
            hover = nominal_cfg.hover_model(run.params)
            reference = output_reference(position_derivatives(run.reference), hover.x_trim)
            if label == OUTPUT_NOMINAL:
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


def gap_summary(rmse_3d: dict[str, np.ndarray], label: str) -> dict | None:
    """Gap closed by `label`, per seed and of the means; None (logged) if true drag is not better."""
    try:
        per_seed = gap_closed(rmse_3d[LTV_NOMINAL], rmse_3d[label], rmse_3d[LTV_TRUE_DRAG])
        of_means = gap_closed(rmse_3d[LTV_NOMINAL].mean(), rmse_3d[label].mean(),
                              rmse_3d[LTV_TRUE_DRAG].mean())
    except ValueError as err:
        logger.warning("gap closed undefined: %s", err)
        return None
    return {"per_seed": per_seed.tolist(), "of_means": float(of_means)}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_run", type=Path, help="identify_outer_loop_model run directory")
    parser.add_argument("nominal", type=Path, help="nominal LTV MPC TOML")
    parser.add_argument("true_drag", type=Path, help="true-drag LTV MPC TOML (ltv_mpc.toml)")
    parser.add_argument("experiments", type=Path, nargs="+")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--greybox", type=Path, default=None,
                        help="fit_greybox_drag run directory; adds 'LTV grey-box'")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    parser.add_argument("--no-show", dest="show", action="store_false")
    args = parser.parse_args(argv)

    koopman = OuterKoopmanModel.load(args.model_run)
    nominal_cfg, true_drag_cfg = load_ltv_mpc(args.nominal), load_ltv_mpc(args.true_drag)
    experiments = [p.resolve() for p in args.experiments]
    check_compatible(koopman, nominal_cfg, true_drag_cfg, load_experiment(experiments[0]).params)
    run_dir = start_run(args.results_dir, "gap_closure", 0,
                        [*nominal_cfg.config_paths, true_drag_cfg.config_path, *experiments],
                        REPO_ROOT)
    seeds = list(range(args.seeds))
    greybox_cfg = (nominal_cfg.with_drag_coeffs(load_greybox_drag(args.greybox))
                   if args.greybox is not None else None)
    summary = {"model_run": str(args.model_run.resolve()), "seeds": seeds,
               "greybox_run": str(args.greybox.resolve()) if args.greybox else None,
               "greybox_drag_coeffs": (greybox_cfg.model_drag_coeffs.tolist()
                                       if greybox_cfg else None)}
    figures = {}

    for path in experiments:
        name = path.stem
        exp = load_experiment(path)   # the models do not depend on the seed: build them once
        derivatives = position_derivatives(exp.reference)
        ltv_cfgs = {LTV_NOMINAL: nominal_cfg, LTV_TRUE_DRAG: true_drag_cfg}
        if greybox_cfg is not None:
            ltv_cfgs[LTV_GREYBOX] = greybox_cfg
        ltvs = {label: (cfg, cfg.make_model(exp.params, derivatives, exp.duration))
                for label, cfg in ltv_cfgs.items()}
        per_seed = [run_controllers(path, seed, koopman, ltvs) for seed in seeds]
        labels = list(per_seed[0])
        errors = {label: np.stack([position_error(logs[label]) for logs in per_seed])
                  for label in labels}
        rmse_axes = {label: rmse(errors[label], axis=1) for label in labels}   # (seeds, 3)
        rmse_3d = {label: np.linalg.norm(v, axis=1) for label, v in rmse_axes.items()}
        gap = {label: gap_summary(rmse_3d, label) for label in (KOOPMAN, LTV_GREYBOX)
               if label in rmse_3d}
        summary[name] = {
            "actuator": exp.actuator.model,
            "rmse_3d_m": {label: {"per_seed": v.tolist(), "mean": float(v.mean()),
                                  "std": float(v.std())} for label, v in rmse_3d.items()},
            "reduction_from_nominal_pct": {
                label: float(percent_reduction(rmse_3d[LTV_NOMINAL].mean(), v.mean()))
                for label, v in rmse_3d.items()},
            "rmse_per_axis_mean_m": {label: v.mean(axis=0).tolist()
                                     for label, v in rmse_axes.items()},
            "gap_closed": gap,
        }
        logger.info("%s (%s rotors): %s | gap closed %s", name, exp.actuator.model,
                    "  ".join(f"{label} {v.mean():.4f}±{v.std():.4f}"
                              for label, v in rmse_3d.items()),
                    "  ".join(f"{label} " + ("undefined" if g is None else f"{g['of_means']:.2f}")
                              for label, g in gap.items()))
        t = per_seed[0][LTV_NOMINAL].t
        figures[f"{name}_tracking_error"] = plot_tracking_error(t, errors, name)
        figures[f"{name}_mse"] = plot_mse(t, errors, baseline=LTV_NOMINAL, title=name)
        figures[f"{name}_top_view_seed0"] = plot_top_view(per_seed[0], f"{name} seed 0")

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
