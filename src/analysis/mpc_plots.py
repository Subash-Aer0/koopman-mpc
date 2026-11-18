"""Plots for outer-loop MPC runs: commands against limits, solver diagnostics, top view.

Works only from logged data (SimulationLog and MPC log arrays), like plotting.py.
Positions are NED; the top view draws north up and east right.
"""
import matplotlib.pyplot as plt
import numpy as np

from controllers.sdlqr_model import POS as REF_POS
from dynamics.dynamics import POS
from simulation.closed_loop import SimulationLog

_COMMAND_LABELS = ("ω_x [rad/s]", "ω_y [rad/s]", "ω_z [rad/s]", "c [m/s²]")


def plot_commands(logs: dict[str, SimulationLog], u_lb, u_ub, title: str = "") -> plt.Figure:
    """Commanded body rates and thrust per controller, with the MPC box limits.

    Args:
        logs: Label -> SimulationLog; u_cmd is plotted.
        u_lb, u_ub: (4,) MPC command bounds, drawn as dashed lines.
        title: Figure title prefix.
    """
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 9))
    for j, (label, log) in enumerate(logs.items()):
        for i, ax in enumerate(axes):
            ax.step(log.t, log.u_cmd[:, i], where="post", color=f"C{j}", label=label)
    for i, ax in enumerate(axes):
        for bound in (u_lb[i], u_ub[i]):
            ax.axhline(bound, color="k", linestyle="--", linewidth=0.8)
        ax.set_ylabel(_COMMAND_LABELS[i])
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{title} commands (dashed: MPC limits)".strip())
    fig.tight_layout()
    return fig


def plot_solver(mpc_logs: dict[str, dict], period: float, title: str = "") -> plt.Figure:
    """Solve time per step against the sample period, and fallback steps.

    Args:
        mpc_logs: Label -> MPCLog.arrays() (keys t, fallback, solve_time, ...).
        period: Controller period, s, drawn as the real-time budget.
    """
    fig, (ax_time, ax_fb) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    for j, (label, data) in enumerate(mpc_logs.items()):
        ax_time.plot(data["t"], 1e3 * data["solve_time"], f"C{j}", label=label)
        ax_fb.step(data["t"], data["fallback"].astype(float) + 0.02 * j, where="post",
                   color=f"C{j}", label=f"{label}: {int(data['fallback'].sum())} fallbacks")
    ax_time.axhline(1e3 * period, color="k", linestyle="--", linewidth=0.8, label="period")
    ax_time.set_yscale("log")
    ax_time.set_ylabel("solve time [ms]")
    ax_time.legend(fontsize=8)
    ax_fb.set_yticks([0, 1], ["solved", "fallback"])
    ax_fb.set_xlabel("time [s]")
    ax_fb.legend(fontsize=8)
    fig.suptitle(f"{title} QP solver".strip())
    fig.tight_layout()
    return fig


def plot_top_view(logs: dict[str, SimulationLog], title: str = "") -> plt.Figure:
    """North–east path of every controller over the reference."""
    fig, ax = plt.subplots(figsize=(7, 7))
    first = next(iter(logs.values()))
    p_ref = first.x_ref[:, REF_POS]
    ax.plot(p_ref[:, 1], p_ref[:, 0], "k--", label="reference")
    for j, (label, log) in enumerate(logs.items()):
        p = log.x[:, POS]
        ax.plot(p[:, 1], p[:, 0], f"C{j}", label=label)
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    ax.set_title(f"{title} top view".strip())
    fig.tight_layout()
    return fig


def position_error(log: SimulationLog) -> np.ndarray:
    """e_p = p − p_ref, world NED, m, shape (T, 3)."""
    return log.x[:, POS] - log.x_ref[:, REF_POS]


def plot_sweep(x, x_label: str, results: dict[str, tuple], title: str = "") -> plt.Figure:
    """Mean ± std of a metric against a swept parameter, one line per controller.

    Args:
        x: Swept values, shape (K,).
        x_label: Axis label, e.g. "peak reference tilt [rad]".
        results: Label -> (mean (K,), std (K,)), e.g. 3-D position RMSE over seeds, m.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, (label, (mean, std)) in enumerate(results.items()):
        ax.errorbar(x, mean, yerr=std, color=f"C{j}", marker="o", capsize=3, label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel("3-D position RMSE [m]")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    return fig
