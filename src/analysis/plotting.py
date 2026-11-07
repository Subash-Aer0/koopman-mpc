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


def plot_prediction(t, true_states, pred_model1, pred_model2, path):
    """Fig. 2: position, velocity, orientation prediction."""
    raise NotImplementedError


def plot_tracking_error(t, errors_by_controller, path):
    """Fig. 3: absolute tracking error per axis, Model 1 vs Model 2."""
    raise NotImplementedError


def plot_mse(t, mse_by_controller, path):
    """Fig. 4: MSE over Monte Carlo runs."""
    raise NotImplementedError
