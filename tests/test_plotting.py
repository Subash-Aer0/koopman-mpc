"""Metrics and per-run tracking plots."""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from analysis.metrics import rmse
from analysis.plotting import (
    plot_mse,
    plot_prediction,
    plot_tracking,
    plot_tracking_error,
    save_figures,
)
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


def test_prediction_figure_skips_nan_rows_in_rmse():
    # An H-step prediction has no value for the first H samples; the RMSE in
    # each panel title must cover only the predicted samples.
    t = np.arange(5) * 0.1
    true = np.zeros((5, 9))
    predicted = np.full((5, 9), 0.2)
    predicted[:2] = np.nan
    fig = plot_prediction(t, true, {"2-step": predicted}, title="test")
    assert fig.axes[0].get_title() == "position error x: RMSE 0.2"
    plt.close(fig)


def monte_carlo_errors():
    """Two runs, four samples: baseline e_x = ±0.3 (|e| 0.3), other e_x = 0.1, 0.2 m."""
    t = np.arange(4) * 0.1
    base = np.zeros((2, 4, 3))
    base[0, :, 0], base[1, :, 0] = 0.3, -0.3
    other = np.zeros((2, 4, 3))
    other[0, :, 0], other[1, :, 0] = 0.1, 0.2
    return t, {"LQR": base, "LQR + K-MPC": other}


def test_abs_error_figure_plots_mean_absolute_error_per_axis():
    t, errors = monte_carlo_errors()
    fig = plot_tracking_error(t, errors, title="test")
    x_axis = fig.axes[0]
    lqr, kmpc = (line.get_ydata() for line in x_axis.get_lines())
    plt.close(fig)
    np.testing.assert_allclose(lqr, 0.3)        # sign does not cancel
    np.testing.assert_allclose(kmpc, 0.15)


def test_mse_figure_plots_mse_and_paired_difference():
    t, errors = monte_carlo_errors()
    fig = plot_mse(t, errors, baseline="LQR", title="test")
    mse_axis, diff_axis = fig.axes
    lqr, kmpc = (line.get_ydata() for line in mse_axis.get_lines())
    diff = [line for line in diff_axis.get_lines() if line.get_label().startswith("LQR + K-MPC")]
    plt.close(fig)
    np.testing.assert_allclose(lqr, 0.09)
    np.testing.assert_allclose(kmpc, (0.01 + 0.04) / 2)
    # Paired: mean of (0.01 − 0.09, 0.04 − 0.09).
    np.testing.assert_allclose(diff[0].get_ydata(), -0.065)


def test_mse_figure_requires_known_baseline():
    t, errors = monte_carlo_errors()
    with pytest.raises(ValueError, match="baseline 'PID'"):
        plot_mse(t, errors, baseline="PID")


@pytest.mark.parametrize("change, match", [
    (lambda t, e: (t[:3], e), "T = len"),
    (lambda t, e: (t, {**e, "LQR": e["LQR"][:1]}), "same"),
    (lambda t, e: (t, {}), "empty"),
    (lambda t, e: (t, {**e, "LQR": np.full((2, 4, 3), np.nan)}), "NaN"),
])
def test_evaluation_figures_reject_bad_errors(change, match):
    t, errors = change(*monte_carlo_errors())
    with pytest.raises(ValueError, match=match):
        plot_tracking_error(t, errors)
