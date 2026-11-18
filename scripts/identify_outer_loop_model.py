"""Fit the outer-loop Koopman model, check excitation, and validate against the physics model.

    PYTHONPATH=src python scripts/identify_outer_loop_model.py configs/identify/outer_loop.toml \
        results/runs/<time>_outer_loop_data

Writes results/runs/<time>_outer_loop_koopman/ with outer_model.{npz,json},
identification.json and figures:
    prediction_rmse.png   open-loop RMSE vs steps ahead per state, Koopman vs hover model,
                          with the measured wind and with only the mean wind (as the MPC)
    spectrum.png          eigenvalues of A (Koopman) and of the hover model
identification.json reports the persistency-of-excitation test (Hankel rank),
regressor rank and conditioning, every λ's validation error, and the slowest
modes with their input and wind gains (PBH).
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plotting import save_figures
from controllers.hover_model import HoverModel
from datagen.outer_loop import load_episodes
from dynamics.params import load_params
from experiment.builders import load_toml
from experiment.ltv_mpc_config import load_ltv_mpc
from experiment.run_record import start_run
from koopman.edmd import build_snapshots, fit_edmd
from koopman.excitation import persistency_of_excitation
from koopman.outer_model import (
    PHYSICAL_DIM, STATE_NAMES, OuterKoopmanModel, deviation_state, horizon_rmse,
)
from koopman.preprocessing import regressor_diagnostics
from koopman.trig_lifting import TrigMonomialLifting
from koopman.structured_edmd import fit_translation_invariant

REPO_ROOT = Path(__file__).resolve().parents[1]
UNITS = ["m"] * 3 + ["m/s"] * 3 + ["rad"] * 3
logger = logging.getLogger("identify_outer_loop_model")


def deviation_episodes(episodes, model: HoverModel, include_rates: bool) -> list[dict]:
    out = []
    for ep in episodes:
        dx = deviation_state(ep.x9, ep.omega, model.x_trim, include_rates)
        out.append({"name": ep.name, "dx": dx, "y": dx[:, :PHYSICAL_DIM],
                    "du": ep.u - model.u_trim, "dw": ep.v_w - model.wind_mean})
    return out


def without_wind(episodes) -> list[dict]:
    return [{**ep, "dw": np.zeros_like(ep["dw"])} for ep in episodes]


def mode_report(A, B, D, names, dt, count=8) -> list[dict]:
    """Slowest modes: eigenvalue, time constant, dominant features, |wᵀB|, |wᵀD| (left eigvecs)."""
    eigvals, right = np.linalg.eig(A)
    left = np.linalg.inv(right)          # rows are left eigenvectors, wᵢᵀ rⱼ = δᵢⱼ
    order = np.argsort(-np.abs(eigvals))
    report = []
    for i in order[:count]:
        w = left[i] / np.linalg.norm(left[i])
        lam = eigvals[i]
        rate = np.log(lam + 0j) / dt
        top = np.argsort(-np.abs(w))[:3]
        report.append({
            "eigenvalue": [float(lam.real), float(lam.imag)],
            "abs": float(abs(lam)),
            "time_constant_s": float(-1 / rate.real) if abs(rate.real) > 1e-12 else None,
            "frequency_hz": float(abs(rate.imag) / (2 * np.pi)),
            "input_gain": float(np.linalg.norm(w @ B)),
            "wind_gain": float(np.linalg.norm(w @ D)),
            "dominant_features": [[names[j], float(abs(w[j]))] for j in top],
        })
    return report


def plot_prediction_rmse(errors: dict, title: str) -> plt.Figure:
    fig, axes = plt.subplots(3, 3, sharex=True, figsize=(13, 9))
    for label, (style, err) in errors.items():
        steps = np.arange(1, len(err) + 1)
        for i, ax in enumerate(axes.T.ravel()):
            ax.plot(steps, err[:, i], style, label=label)
    for i, ax in enumerate(axes.T.ravel()):
        ax.set_title(f"{STATE_NAMES[i]} [{UNITS[i]}]", fontsize=10)
        ax.set_ylim(bottom=0)
    for ax in axes[-1]:
        ax.set_xlabel("steps ahead (0.1 s)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_spectrum(koopman_eigs, hover_eigs) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "k", linewidth=0.6)
    ax.plot(koopman_eigs.real, koopman_eigs.imag, "C0o", markersize=5, label="Koopman A")
    ax.plot(hover_eigs.real, hover_eigs.imag, "C3x", markersize=8, label="hover model A")
    ax.set_aspect("equal")
    ax.set_xlabel("Re λ")
    ax.set_ylabel("Im λ")
    ax.legend()
    ax.set_title("discrete eigenvalues (0.1 s)")
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("dataset", type=Path, help="collect_outer_loop_data run directory")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    parser.add_argument("--no-show", dest="show", action="store_false")
    args = parser.parse_args(argv)

    path = args.config.resolve()
    section = load_toml(path)["identify"]
    mpc_path = (path.parent / section["mpc_config"]).resolve()
    dataset_dir = args.dataset.resolve()
    meta = json.loads((dataset_dir / "metadata.json").read_text())
    vehicle_path = next(Path(p) for p in meta["configs"] if Path(p).name == "quadrotor.toml")
    mpc_cfg = load_ltv_mpc(mpc_path)
    params = load_params(vehicle_path)
    # Coordinates are deviations from the trim of the model the MPC config
    # believes in (nominal: level hover, c = g), so no oracle knowledge of the
    # drag enters the Koopman model. It is also the physics model compared against.
    hover = mpc_cfg.hover_model(params)
    sample_time = mpc_cfg.base.sample_time
    include_rates = bool(section["include_body_rates"])
    horizon = int(section["horizon"])
    invariant = bool(section.get("translation_invariant", False))
    run_dir = start_run(args.results_dir, section["name"], int(section["seed"]),
                        [path, *mpc_cfg.config_paths], REPO_ROOT)

    splits = load_episodes(dataset_dir)
    train = deviation_episodes(splits["train"], hover, include_rates)
    valid = deviation_episodes(splits["validation"], hover, include_rates)
    state_dim = train[0]["dx"].shape[1]
    names = STATE_NAMES[:state_dim]
    nonlinear = [names.index(n) for n in section["nonlinear_states"]]
    trig = [names.index(n) for n in section.get("trig_angles", [])]
    scale = np.concatenate([ep["dx"] for ep in train]).std(axis=0)
    if np.any(scale < 1e-9):
        raise ValueError(f"unexcited state channels: {np.array(names)[scale < 1e-9].tolist()}")
    lifting = TrigMonomialLifting(state_dim, int(section["degree"]), nonlinear, trig,
                                  scale[trig] if trig else None)

    # Question 1: is the input rich enough? Hankel rank (PE), then the lifted regressor.
    n_lift, m, d = lifting.lifted_dim, 4, 3
    pe = {
        f"inputs_order_n+1={state_dim + 1}":
            persistency_of_excitation([ep["du"] for ep in train], state_dim + 1),
        f"inputs_and_wind_order_n+1={state_dim + 1}":
            persistency_of_excitation([np.hstack([ep["du"], ep["dw"]]) for ep in train],
                                      state_dim + 1),
        f"inputs_order_N+1={n_lift + 1}":
            persistency_of_excitation([ep["du"] for ep in train], n_lift + 1),
    }
    for key, value in pe.items():
        logger.info("PE %-32s rank %d/%d  σmin/σmax %.3g  -> %s", key, value["rank"],
                    value["rows"], value["sigma_min_rel"],
                    "persistently exciting" if value["persistently_exciting"] else "NOT PE")
    gamma1, gamma2 = build_snapshots([ep["dx"] / scale for ep in train],
                                     [ep["du"] for ep in train], [ep["dw"] for ep in train],
                                     lifting)
    regressor = regressor_diagnostics(gamma1, n_lift + m + d)
    logger.info("lifted regressor: rank %d/%d, %d snapshots, condition number %.3g",
                regressor["rank"], regressor["rows"], regressor["columns"],
                regressor["condition_number"])

    # Fit every λ; keep the lowest mean validation RMSE (normalised per state) at `horizon`.
    hover_err = horizon_rmse(hover.A, hover.B, hover.D, np.eye(PHYSICAL_DIM), lambda dx: dx,
                             [{**ep, "dx": ep["y"]} for ep in valid], horizon)
    candidates, failed = {}, {}
    for lam in section["l1_lambdas"]:
        try:
            if invariant:
                if float(lam) != 0.0:
                    raise RuntimeError("the translation-invariant fit is least squares only")
                A, B, D = fit_translation_invariant(gamma1, gamma2, n_lift, m, d, [0, 1, 2],
                                                    lifting.constant_index)
            else:
                A, B, D = fit_edmd(gamma1, gamma2, n_lift, m, d, float(lam),
                                   lifting.constant_index)
        except RuntimeError as err:   # solver cap or unsupported option; reported, not hidden
            failed[str(lam)] = str(err)
            logger.warning("λ = %g not fitted: %s", lam, err)
            continue
        model = OuterKoopmanModel(A=A, B=B, D=D, scale=scale, degree=lifting.degree,
                                  nonlinear_indices=lifting.nonlinear_indices,
                                  trig_indices=lifting.trig_indices,
                                  include_rates=include_rates, x_trim=hover.x_trim,
                                  u_trim=hover.u_trim, wind_mean=hover.wind_mean,
                                  dt=sample_time)
        err = horizon_rmse(A, B, D, model.C, model.lift_deviation, valid, horizon)
        candidates[float(lam)] = (model, err, float(np.mean(err[-1] / hover_err[-1])))
        logger.info("λ = %-6g validation RMSE at %d steps / hover model's, mean over states: %.3f",
                    lam, horizon, candidates[float(lam)][2])
    if not candidates:
        raise RuntimeError(f"no λ could be fitted: {failed}")
    best = min(candidates, key=lambda lam: candidates[lam][2])
    model, koop_err, _ = candidates[best]
    model.save(run_dir)

    koop_mean = horizon_rmse(model.A, model.B, model.D, model.C, model.lift_deviation,
                             without_wind(valid), horizon)
    hover_mean = horizon_rmse(hover.A, hover.B, hover.D, np.eye(PHYSICAL_DIM), lambda dx: dx,
                              without_wind([{**ep, "dx": ep["y"]} for ep in valid]), horizon)
    per_episode = {}
    for ep in valid:
        k = horizon_rmse(model.A, model.B, model.D, model.C, model.lift_deviation,
                         without_wind([ep]), horizon)[-1]
        h = horizon_rmse(hover.A, hover.B, hover.D, np.eye(PHYSICAL_DIM), lambda dx: dx,
                         without_wind([{**ep, "dx": ep["y"]}]), horizon)[-1]
        per_episode[ep["name"]] = {"koopman": k.tolist(), "hover": h.tolist()}

    feature_names = lifting.feature_names(list(names))
    report = {
        "dataset": str(dataset_dir), "state_names": list(names), "lifted_dim": n_lift,
        "translation_invariant": invariant,
        "features": feature_names, "scale": scale.tolist(), "selected_l1_lambda": best,
        "persistency_of_excitation": pe, "regressor": regressor,
        "validation_score_by_lambda": {str(k): v[2] for k, v in candidates.items()},
        "lambdas_not_fitted": failed,
        f"rmse_at_{horizon}_steps_measured_wind": {
            "koopman": koop_err[-1].tolist(), "hover": hover_err[-1].tolist()},
        f"rmse_at_{horizon}_steps_mean_wind": {
            "koopman": koop_mean[-1].tolist(), "hover": hover_mean[-1].tolist()},
        "rmse_at_1_step_mean_wind": {"koopman": koop_mean[0].tolist(),
                                    "hover": hover_mean[0].tolist()},
        f"rmse_at_{horizon}_steps_mean_wind_per_episode": per_episode,
        "units": UNITS,
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(model.A)))),
        "slowest_modes": mode_report(model.A, model.B, model.D, feature_names, model.dt),
    }
    (run_dir / "identification.json").write_text(json.dumps(report, indent=2))
    for i, name in enumerate(STATE_NAMES[:PHYSICAL_DIM]):
        logger.info("%-6s RMSE at %d steps (mean wind): Koopman %.4f  hover model %.4f %s",
                    name, horizon, koop_mean[-1, i], hover_mean[-1, i], UNITS[i])

    figures = {
        "prediction_rmse": plot_prediction_rmse({
            "Koopman, measured wind": ("C0-", koop_err),
            "hover model, measured wind": ("C3-", hover_err),
            "Koopman, mean wind": ("C0--", koop_mean),
            "hover model, mean wind": ("C3--", hover_mean),
        }, f"open-loop prediction RMSE on validation flights (λ = {best:g})"),
        "spectrum": plot_spectrum(np.linalg.eigvals(model.A), np.linalg.eigvals(hover.A)),
    }
    save_figures(figures, run_dir / "figures")
    logger.info("model written to %s", run_dir)
    if args.show:
        plt.show()
    plt.close("all")
    return run_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    main()
