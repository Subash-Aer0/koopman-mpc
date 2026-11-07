"""Metrics and per-run tracking plots."""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from analysis.metrics import rmse
from analysis.plotting import plot_tracking, save_figures
from simulation.closed_loop import SimulationLog, Timing


def make_log(n=50) -> SimulationLog:
    t = np.arange(n) * 0.1
    x_ref = np.zeros((n, 10))
    x_ref[:, 0] = np.sin(t)
    x_ref[:, 3] = 1.0
    x = np.zeros((n, 13))
    x[:, :10] = x_ref
    x[:, 0] += 0.2  # constant 0.2 m north error
    zeros4 = np.zeros((n, 4))
    return SimulationLog(t=t, x=x, x_ref=x_ref, u_ref=zeros4, u_h=zeros4, u_d=zeros4,
                         u_cmd=zeros4, v_w=np.zeros((n, 3)), timing=Timing())


def test_rmse_known_values():
    np.testing.assert_allclose(rmse([[3.0, 0.0], [-3.0, 4.0]]), [3.0, np.sqrt(8.0)])


def test_rmse_rejects_nan():
    with pytest.raises(ValueError):
        rmse([1.0, np.nan])


def test_plot_tracking_saves_both_figures(tmp_path):
    figures = plot_tracking(make_log(), title="test")
    paths = save_figures(figures, tmp_path / "figures")
    plt.close("all")
    assert sorted(p.name for p in paths) == ["tracking.png", "trajectory_3d.png"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_tracking_figure_reports_per_axis_rmse():
    figures = plot_tracking(make_log())
    titles = [ax.get_title() for ax in figures["tracking"].axes[1::2]]
    plt.close("all")
    assert titles == ["RMSE 0.200 m", "RMSE 0.000 m", "RMSE 0.000 m"]
