"""Figures 2-4, and per-run tracking plots.

Works only from logged data (SimulationLog), never from runtime objects, so
any saved run can be re-plotted. Positions are NED; 3-D plots show
altitude = −z so that up is drawn up.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from controllers.sdlqr_model import POS as REF_POS
from dynamics.dynamics import POS
from simulation.closed_loop import SimulationLog

from .metrics import rmse

_AXIS_LABELS = ("x (north)", "y (east)", "z (down)")


def plot_tracking(log: SimulationLog, title: str = "") -> dict[str, plt.Figure]:
    """Reference vs tracked position and per-axis tracking error.

    Error is e = p − p_ref, NED world frame, m (same sign as the LQR's x − x_ref).

    Args:
        log: Logged run.
        title: Figure title prefix, e.g. the run name.

    Returns:
        {"trajectory_3d": 3-D path figure, "tracking": 3x2 time-series figure},
        keyed by the file stem used by save_figures.
    """
    t = log.t
    p = log.x[:, POS]
    p_ref = log.x_ref[:, REF_POS]
    err = p - p_ref
    axis_rmse = rmse(err, axis=0)

    fig_3d = plt.figure(figsize=(8, 6))
    ax = fig_3d.add_subplot(projection="3d")
    ax.plot(p_ref[:, 0], p_ref[:, 1], -p_ref[:, 2], "k--", label="reference")
    ax.plot(p[:, 0], p[:, 1], -p[:, 2], "C0", label="tracked")
    ax.scatter(*p[0, :2], -p[0, 2], color="C2", label="start")
    ax.set_xlabel("north [m]")
    ax.set_ylabel("east [m]")
    ax.set_zlabel("altitude −z [m]")
    # Equal metric scale on all axes; a flat axis (e.g. hover) keeps 10% of the
    # largest span so it stays visible.
    spans = np.ptp(np.vstack([p, p_ref]), axis=0)
    ax.set_box_aspect(np.maximum(spans, 0.1 * max(spans.max(), 1e-9)))
    ax.legend()
    ax.set_title(f"{title} 3-D position (3-D RMSE {np.linalg.norm(axis_rmse):.3f} m)".strip())

    fig_ts, axes = plt.subplots(3, 2, sharex=True, figsize=(12, 8))
    for i, label in enumerate(_AXIS_LABELS):
        axes[i, 0].plot(t, p_ref[:, i], "k--", label="reference")
        axes[i, 0].plot(t, p[:, i], "C0", label="tracked")
        axes[i, 0].set_ylabel(f"{label} [m]")
        axes[i, 1].plot(t, err[:, i], "C3")
        axes[i, 1].axhline(0.0, color="k", linewidth=0.5)
        axes[i, 1].set_ylabel(f"error {label[0]} [m]")
        axes[i, 1].set_title(f"RMSE {axis_rmse[i]:.3f} m", fontsize=9)
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].set_title("position")
    for ax_bottom in axes[-1]:
        ax_bottom.set_xlabel("time [s]")
    fig_ts.suptitle(f"{title} tracking: p and e = p − p_ref (NED)".strip())
    fig_ts.tight_layout()

    return {"trajectory_3d": fig_3d, "tracking": fig_ts}


def save_figures(figures: dict[str, plt.Figure], directory: str | Path) -> list[Path]:
    """Save each figure as <directory>/<name>.png; returns the written paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figures.items():
        path = directory / f"{name}.png"
        fig.savefig(path, dpi=150)
        paths.append(path)
    return paths


_ERROR_GROUPS = (("position error", "m", ("x", "y", "z")),
                 ("velocity error", "m/s", ("x", "y", "z")),
                 ("attitude error", "rad", ("roll", "pitch", "yaw")))


def plot_prediction(t, true_states, predictions: dict[str, np.ndarray],
                    title: str = "") -> plt.Figure:
    """Fig. 2: true vs predicted tracking error e_x, one panel per channel.

    Args:
        t: Sample times, s, shape (T,).
        true_states: Logged e_x = [p, v, rpy] error, shape (T, 9); m, m/s, rad.
        predictions: Label -> predicted e_x, shape (T, 9) each, e.g. {"Model 1": ...};
            NaN rows (no prediction yet, e.g. the first H samples of an H-step
            prediction) are left blank.
        title: Figure title.

    Returns:
        3x3 figure: rows position / velocity / attitude, columns the three axes;
        each panel is titled with the prediction RMSE of the first model over
        the samples where it has a prediction.
    """
    true_states = np.asarray(true_states)
    fig, axes = plt.subplots(3, 3, sharex=True, figsize=(13, 8))
    for row, (group, unit, axis_names) in enumerate(_ERROR_GROUPS):
        for col, axis_name in enumerate(axis_names):
            i = 3 * row + col
            ax = axes[row, col]
            ax.plot(t, true_states[:, i], "k", label="true state")
            for j, (label, predicted) in enumerate(predictions.items()):
                ax.plot(t, predicted[:, i], f"C{j}--", label=label)
            ax.axhline(0.0, color="k", linewidth=0.5)
            ax.set_ylabel(f"{axis_name} [{unit}]")
            if predictions:
                first = np.asarray(next(iter(predictions.values())))[:, i]
                valid = np.isfinite(first)
                error = rmse(first[valid] - true_states[valid, i])
                ax.set_title(f"{group} {axis_name}: RMSE {error:.3g}", fontsize=9)
    axes[0, 0].legend(loc="best", fontsize=8)
    for ax_bottom in axes[-1]:
        ax_bottom.set_xlabel("time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


_BAND = (10, 90)  # percentiles across runs shown as a shaded band


def _check_errors(t, errors_by_controller) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Validate {label: e_p of shape (runs, T, 3)} against t (T,); all share runs and T."""
    t = np.asarray(t, dtype=float)
    if not errors_by_controller:
        raise ValueError("errors_by_controller is empty")
    errors = {label: np.asarray(e, dtype=float) for label, e in errors_by_controller.items()}
    shapes = {e.shape for e in errors.values()}
    if len(shapes) != 1:
        raise ValueError(f"all controllers need the same (runs, T, 3) shape, got {shapes}")
    shape = shapes.pop()
    if len(shape) != 3 or shape[2] != 3 or t.shape != (shape[1],):
        raise ValueError(f"errors must be (runs, T, 3) with T = len(t) = {len(t)}, got {shape}")
    if not all(np.all(np.isfinite(e)) for e in errors.values()):
        raise ValueError("errors contain NaN or inf")
    return t, errors


def plot_tracking_error(t, errors_by_controller: dict, title: str = "") -> plt.Figure:
    """Fig. 3: absolute position tracking error per axis over time, across Monte Carlo runs.

    Args:
        t: Sample times, s, shape (T,).
        errors_by_controller: Label -> e_p = p − p_ref, NED world frame, m,
            shape (runs, T, 3); the same runs (wind seeds) for every label.
        title: Figure title.

    Returns:
        3x1 figure: per axis, the mean |e| over runs as a line and the
        10th-90th percentile of |e| across runs as a band.
    """
    t, errors = _check_errors(t, errors_by_controller)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    for i, (ax, label) in enumerate(zip(axes, _AXIS_LABELS)):
        for j, (name, e) in enumerate(errors.items()):
            abs_e = np.abs(e[:, :, i])
            ax.plot(t, abs_e.mean(axis=0), f"C{j}", label=name)
            ax.fill_between(t, *np.percentile(abs_e, _BAND, axis=0), color=f"C{j}", alpha=0.2,
                            linewidth=0)
        ax.set_ylabel(f"|error {label}| [m]")
        ax.set_ylim(bottom=0.0)
    n_runs = next(iter(errors.values())).shape[0]
    axes[0].legend(loc="upper right", title=f"mean, {_BAND[0]}-{_BAND[1]}% band, {n_runs} runs",
                   fontsize=8, title_fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{title} absolute tracking error (NED)".strip())
    fig.tight_layout()
    return fig


def plot_mse(t, errors_by_controller: dict, baseline: str, title: str = "") -> plt.Figure:
    """Fig. 4: position MSE over time across Monte Carlo runs, and the paired difference.

    MSE(t) = mean over runs of ‖e_p(t)‖², m². The lower panel shows, for each
    non-baseline label, the per-run difference ‖e_p(t)‖² − ‖e_p,baseline(t)‖²
    on the same wind seed (mean and 10th-90th percentile): below zero means
    better than the baseline at that time, with wind luck removed.

    Args:
        t: Sample times, s, shape (T,).
        errors_by_controller: Label -> e_p, m, shape (runs, T, 3), same runs per label.
        baseline: Label the differences are taken against.
        title: Figure title.
    """
    t, errors = _check_errors(t, errors_by_controller)
    if baseline not in errors:
        raise ValueError(f"baseline '{baseline}' is not one of {list(errors)}")
    squared = {name: np.sum(e**2, axis=2) for name, e in errors.items()}  # (runs, T)

    fig, (ax_mse, ax_diff) = plt.subplots(2, 1, sharex=True, figsize=(10, 7),
                                          gridspec_kw={"height_ratios": [2, 1]})
    for j, (name, sq) in enumerate(squared.items()):
        ax_mse.plot(t, sq.mean(axis=0), f"C{j}", label=name)
        if name == baseline:
            continue
        diff = sq - squared[baseline]
        ax_diff.plot(t, diff.mean(axis=0), f"C{j}", label=f"{name} − {baseline}")
        ax_diff.fill_between(t, *np.percentile(diff, _BAND, axis=0), color=f"C{j}", alpha=0.2,
                             linewidth=0)
    ax_mse.set_ylabel("MSE ‖e_p‖² [m²]")
    ax_mse.set_ylim(bottom=0.0)
    ax_mse.legend(loc="upper right", fontsize=8)
    ax_diff.axhline(0.0, color="k", linewidth=0.5)
    ax_diff.set_ylabel("paired Δ‖e_p‖² [m²]")
    ax_diff.legend(loc="upper right", fontsize=8,
                   title=f"mean, {_BAND[0]}-{_BAND[1]}% band", title_fontsize=8)
    ax_diff.set_xlabel("time [s]")
    n_runs = next(iter(errors.values())).shape[0]
    fig.suptitle(f"{title} position MSE over {n_runs} paired runs".strip())
    fig.tight_layout()
    return fig
