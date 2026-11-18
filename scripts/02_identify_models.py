"""Fit Koopman Model 1 with l1-EDMD on a datagen dataset; validate by multi-step rollout.

    PYTHONPATH=src python scripts/02_identify_models.py results/runs/<datagen run> \
        configs/identify/model1.toml

Writes results/runs/<time>_<name>/ with metadata, the config copy and:
    model.npz            KoopmanModel.save: A, B, D, C
    standardizer.npz     mean, std of e_x; the model acts on z = (e_x − mean) / std
    identification.json  dataset provenance, N, Γ1 rank / condition number,
                         validation H-step and full-rollout errors per channel
    validation_prediction.npz  true e_x, full-rollout and 1- / H-step predictions
    figures/prediction_<reference>_<repeat>_<mode>.png  Fig. 2-style true vs
                         predicted e_x; mode is 1step, <H>step or rollout

Pipeline (Sec. III-B, IV-B): standardize e_x with training statistics, lift
with ψ, stack snapshot pairs inside each episode, check Γ1 has full row rank,
solve Eq. (8), then validate two ways on each validation episode, driven by
the logged u_d and v_w: H-step predictions re-lifted from the logged state at
every sample (how the MPC uses the model, Sec. IV-C), and a full rollout from
the first sample only (Fig. 2).
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plotting import plot_prediction, save_figures
from datagen.collect import ERROR_DIM, Split
from datagen.dataset import load_dataset
from experiment.builders import load_identify
from experiment.run_record import start_run
from koopman.edmd import build_snapshots, fit_edmd
from koopman.lifting import MonomialLifting
from koopman.models import KoopmanModel
from koopman.preprocessing import Standardizer, regressor_diagnostics

REPO_ROOT = Path(__file__).resolve().parents[1]
ERROR_LABELS = ("p_x", "p_y", "p_z", "v_x", "v_y", "v_z", "roll", "pitch", "yaw")
logger = logging.getLogger("identify_models")


def episode_data(split: Split, standardizer: Standardizer):
    """Per-episode lists (z, u_d, v_w) in the form build_snapshots expects.

    z is the standardized error, shapes (T, n); u_d (T, m) and v_w (T, d) stay physical.
    """
    return (
        [standardizer.transform(e) for e in split.e_x],
        list(split.u_d),
        list(split.v_w),
    )


def projection(lifting, state_dim: int) -> np.ndarray:
    """C = [I_n 0], so z = C ψ(z); checked against the lifting, not assumed.

    Raises:
        ValueError: If ψ(z)[:n] != z, i.e. the lifting breaks its contract.
    """
    C = np.zeros((state_dim, lifting.lifted_dim))
    C[:, :state_dim] = np.eye(state_dim)
    probe = np.random.default_rng(0).standard_normal((4, state_dim))
    if not np.allclose(lifting(probe) @ C.T, probe):
        raise ValueError("lifting must place the state first: ψ(z)[:n] == z")
    return C


def rollout(model: KoopmanModel, lifting, C: np.ndarray, z0, u_d, v_w) -> np.ndarray:
    """Open-loop prediction from z0 alone: ψ_0 = ψ(z0), ψ_k+1 = model.predict(ψ_k, ...).

    Args:
        z0: Initial standardized error, shape (n,).
        u_d: Inputs, shape (T, m); u_d[k] drives step k -> k+1.
        v_w: Wind, shape (T, d); v_w[k] drives step k -> k+1.

    Returns:
        Predicted z, shape (T, n), z_pred[0] == z0.
    """
    psi = lifting(np.asarray(z0, dtype=float)[None, :])[0]
    z_pred = np.empty((len(u_d), len(z0)))
    z_pred[0] = C @ psi
    for k in range(len(u_d) - 1):
        psi = model.predict(psi, u_d[k], v_w[k])
        z_pred[k + 1] = C @ psi
    if not np.all(np.isfinite(z_pred)):
        raise ValueError(
            "rollout diverged to NaN or inf; the identified A is likely unstable"
        )
    return z_pred


def predict_split(
    model: KoopmanModel,
    lifting,
    C: np.ndarray,
    standardizer: Standardizer,
    split: Split,
) -> np.ndarray:
    """Rollout of every episode from its first sample; predicted e_x, shape (E, T, 9)."""
    z, u_d, v_w = episode_data(split, standardizer)
    return np.stack(
        [
            standardizer.inverse(rollout(model, lifting, C, z_i[0], u_i, w_i))
            for z_i, u_i, w_i in zip(z, u_d, v_w)
        ]
    )


def rollout_errors(predicted: np.ndarray, true: np.ndarray) -> dict:
    """Multi-step rollout error per channel over all episodes, physical units.

    nrmse = rmse / rms(e_x): 0 is perfect; 1 is no better than predicting e_x = 0.
    """
    rmse = np.sqrt(np.mean((predicted - true) ** 2, axis=(0, 1)))
    rms = np.sqrt(np.mean(true**2, axis=(0, 1)))
    return {
        label: {"rmse": float(r), "nrmse": float(r / s)}
        for label, r, s in zip(ERROR_LABELS, rmse, rms)
    }


def horizon_predictions(
    model: KoopmanModel,
    lifting,
    C: np.ndarray,
    standardizer: Standardizer,
    split: Split,
    horizon: int,
) -> np.ndarray:
    """H-step predictions, re-lifted from the logged state at every start.

    This is how the MPC uses the model: at each control step it lifts the
    measured e_x and predicts H steps ahead with the planned u_d and the wind.
    Sample k holds the prediction made at k − H from the logged e_x[k − H],
    driven by u_d and v_w over [k − H, k); the first H samples have no
    prediction and are NaN.

    Returns:
        Predicted e_x, shape (E, T, 9), physical units, aligned with split.e_x.

    Raises:
        ValueError: If horizon < 1 or no episode is longer than horizon.
    """
    n_samples = len(split.t)
    if not 1 <= horizon < n_samples:
        raise ValueError(f"horizon must be in [1, {n_samples - 1}], got {horizon}")
    z, u_d, v_w = episode_data(split, standardizer)
    predicted = np.full(split.e_x.shape, np.nan)
    for i, (z_i, u_i, w_i) in enumerate(zip(z, u_d, v_w)):
        psi = lifting(z_i)[: n_samples - horizon]  # one start per row
        for j in range(horizon):
            psi = (
                psi @ model.A.T
                + u_i[j : n_samples - horizon + j] @ model.B.T
                + w_i[j : n_samples - horizon + j] @ model.D.T
            )
        predicted[i, horizon:] = standardizer.inverse(psi @ C.T)
    return predicted


def horizon_errors(
    model: KoopmanModel,
    lifting,
    C: np.ndarray,
    standardizer: Standardizer,
    split: Split,
    horizon: int,
) -> dict:
    """H-step prediction error per channel, pooled over all starts and episodes.

    nrmse as rollout_errors; see horizon_predictions for how predictions are made.
    """
    predicted = horizon_predictions(model, lifting, C, standardizer, split, horizon)
    return rollout_errors(predicted[:, horizon:], split.e_x[:, horizon:])


def main(argv: list[str] | None = None) -> Path:
    """Fit, validate and save Model 1; return the run directory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "dataset", type=Path, help="datagen run directory with dataset.npz"
    )
    parser.add_argument("config", type=Path, help="identify TOML file")
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "results" / "runs"
    )
    args = parser.parse_args(argv)

    cfg = load_identify(args.config)
    splits = load_dataset(args.dataset)
    for name in ("train", "validation"):
        if name not in splits:
            raise ValueError(
                f"{args.dataset} has no '{name}' split; found {sorted(splits)}"
            )
    dataset_metadata = json.loads((args.dataset / "metadata.json").read_text())

    run_dir = start_run(
        args.results_dir, cfg.name, cfg.seed, cfg.config_paths, REPO_ROOT
    )
    logger.info(
        "run %s: p = %d, λ = %g, dataset %s",
        run_dir.name,
        cfg.degree,
        cfg.l1_lambda,
        args.dataset.name,
    )

    train = splits["train"]
    standardizer = Standardizer.fit(train.e_x)
    lifting = MonomialLifting(ERROR_DIM, cfg.degree)
    C = projection(lifting, ERROR_DIM)
    input_dim, dist_dim = train.u_d.shape[-1], train.v_w.shape[-1]

    gamma1, gamma2 = build_snapshots(*episode_data(train, standardizer), lifting)
    diagnostics = regressor_diagnostics(
        gamma1, lifting.lifted_dim + input_dim + dist_dim
    )
    logger.info(
        "Γ1: %d x %d, rank %d, condition number %.3g",
        diagnostics["rows"],
        diagnostics["columns"],
        diagnostics["rank"],
        diagnostics["condition_number"],
    )

    A, B, D = fit_edmd(
        gamma1, gamma2, lifting.lifted_dim, input_dim, dist_dim, cfg.l1_lambda,
        constant_index=lifting.constant_index,
    )
    model = KoopmanModel(A=A, B=B, D=D, C=C, use_residual=False)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    logger.info("spectral radius of A: %.4f", spectral_radius)

    validation = splits["validation"]
    predicted = predict_split(model, lifting, C, standardizer, validation)
    errors = rollout_errors(predicted, validation.e_x)
    # 1 step shows the fit itself; cfg.horizon is what the MPC relies on.
    horizons = sorted({1, cfg.horizon})
    stepped = {
        h: horizon_predictions(model, lifting, C, standardizer, validation, h)
        for h in horizons
    }
    stepped_errors = {
        h: rollout_errors(pred[:, h:], validation.e_x[:, h:])
        for h, pred in stepped.items()
    }
    horizon = stepped_errors[cfg.horizon]
    columns = [f"{h}-step" for h in horizons] + ["full rollout"]
    tables = [stepped_errors[h] for h in horizons] + [errors]
    logger.info("validation nrmse, %s:", " | ".join(columns))
    for label in ERROR_LABELS:
        logger.info(
            "  %-5s %s",
            label,
            " | ".join(f"{table[label]['nrmse']:.3f}" for table in tables),
        )
    logger.info(
        "  mean  %s",
        " | ".join(
            f"{np.mean([err['nrmse'] for err in table.values()]):.3f}"
            for table in tables
        ),
    )

    model.save(run_dir / "model.npz")
    np.savez(run_dir / "standardizer.npz", mean=standardizer.mean, std=standardizer.std)
    np.savez(
        run_dir / "validation_prediction.npz",
        t=validation.t,
        reference=validation.reference,
        true=validation.e_x,
        predicted=predicted,
        **{f"predicted_{h}step": pred for h, pred in stepped.items()},
    )
    modes = {f"{h}step": (f"{h}-step prediction", pred) for h, pred in stepped.items()}
    modes["rollout"] = ("full rollout from t = 0", predicted)
    figures = {}
    for i, (reference, repeat) in enumerate(
        zip(validation.reference, validation.repeat)
    ):
        for mode, (description, pred) in modes.items():
            figures[f"prediction_{reference}_{repeat}_{mode}"] = plot_prediction(
                validation.t,
                validation.e_x[i],
                {f"Koopman Model 1, {description}": pred[i]},
                title=f"{cfg.name}: validation {description}, {reference} #{repeat} "
                f"(p = {cfg.degree}, λ = {cfg.l1_lambda:g})",
            )
    save_figures(figures, run_dir / "figures")
    # plt.close("all")
    (run_dir / "identification.json").write_text(
        json.dumps(
            {
                "dataset": str(args.dataset.resolve()),
                "dataset_git_commit": dataset_metadata["git_commit"],
                "dataset_seed": dataset_metadata["seed"],
                "state_dim": ERROR_DIM,
                "lifted_dim": lifting.lifted_dim,
                "degree": cfg.degree,
                "l1_lambda": cfg.l1_lambda,
                "regressor": diagnostics,
                "spectral_radius_A": spectral_radius,
                "validation_one_step": stepped_errors[1],
                "validation_horizon_steps": cfg.horizon,
                "validation_horizon": horizon,
                "validation_rollout": errors,
            },
            indent=2,
        )
    )
    logger.info("saved model to %s", run_dir)
    return run_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    main()
