"""Hover linear MPC: terminal set, LQR equivalence, terminal modes, reference consistency."""
import numpy as np
import pytest

from controllers.hover_model import EULER, HoverModel, POS, VEL
from controllers.linear_mpc import (
    LinearizedReference, LinearMPC, LinearMPCSettings, outer_state,
)
from controllers.terminal_set import maximal_admissible_set
from datagen.trajectories import CircleReference
from dynamics import dynamics as rigid_body
from dynamics.params import QuadParams
from dynamics.quaternion import Quaternion

WIND = np.array([2.7032, 1.7566, 0.0124])
DT = 0.1


@pytest.fixture
def vehicle() -> QuadParams:
    return QuadParams(mass=1.0, inertia=np.diag([0.0082, 0.0082, 0.0149]),
                      drag_coeffs=np.array([0.05, 0.05, 0.10]), gravity=9.81)


@pytest.fixture
def model(vehicle) -> HoverModel:
    return HoverModel.build(vehicle, WIND, yaw=0.0, dt=DT)


def settings(terminal="polytope", horizon=20) -> LinearMPCSettings:
    rate = 2.0 / np.sqrt(3.0)
    return LinearMPCSettings(
        Q=np.diag([100.0] * 3 + [10.0] * 3 + [1.0] * 3), R=np.diag([1.0, 5.0, 5.0, 0.1]),
        horizon=horizon, terminal=terminal, u_lb=np.array([-rate] * 3 + [2.0]),
        u_ub=np.array([rate] * 3 + [18.0]), max_tilt=0.5)


def setpoint(position):
    position = np.asarray(position, dtype=float)
    return lambda t: (position, np.zeros(3), np.zeros(3), np.zeros(3))


def full_state(x9) -> np.ndarray:
    """9-state [p, v, η] -> 13-state with zero body rates."""
    x = np.zeros(rigid_body.STATE_DIM)
    x[rigid_body.POS] = x9[POS]
    x[rigid_body.VEL] = x9[VEL]
    x[rigid_body.QUAT] = Quaternion.from_euler(x9[EULER]).to_numpy()
    return x


def make_mpc(model, terminal="polytope", horizon=20, derivatives=None, duration=5.0):
    reference = LinearizedReference(model, derivatives or setpoint([0.0, 0.0, 0.0]))
    return LinearMPC(model, settings(terminal, horizon), reference, duration)


def test_terminal_set_of_the_lecture_example_is_unit_interval():
    # [LN-MPC §3.1]: x⁺ = 1.5 x + u, K = −1, |u| ≤ 1, |x| ≤ 3  ->  O∞ = [−1, 1].
    H = np.array([[1.0], [-1.0], [-1.0], [1.0]])   # x ≤ 3, −x ≤ 3, K x ≤ 1, −K x ≤ 1
    h = np.array([3.0, 3.0, 1.0, 1.0])
    H_f, h_f = maximal_admissible_set(np.array([[0.5]]), H, h)
    upper = np.min(h_f[H_f[:, 0] > 0] / H_f[H_f[:, 0] > 0, 0])
    lower = np.max(h_f[H_f[:, 0] < 0] / H_f[H_f[:, 0] < 0, 0])
    assert (lower, upper) == pytest.approx((-1.0, 1.0))


def test_hover_terminal_set_is_invariant_and_admissible(model):
    from scipy.optimize import linprog
    mpc = make_mpc(model)
    H_f, h_f = mpc.terminal_set
    A_cl = model.A + model.B @ mpc.K
    free = [(None, None)] * 9
    for row, bound in zip(H_f @ A_cl, h_f):   # (A + BK) O∞ ⊆ O∞
        result = linprog(-row, A_ub=H_f, b_ub=h_f, bounds=free, method="highs")
        assert result.status == 0 and -result.fun <= bound + 1e-7
    for row, bound in zip(mpc.K, mpc._delta_ub):  # K e within the tightened input bounds
        result = linprog(-row, A_ub=H_f, b_ub=h_f, bounds=free, method="highs")
        assert -result.fun <= bound + 1e-7


def test_cost_only_mpc_equals_lqr_when_no_constraint_is_active(model):
    mpc = make_mpc(model, terminal="cost_only")
    e0 = np.array([0.05, -0.04, 0.03, 0.02, 0.0, -0.01, 0.01, -0.01, 0.02])
    u = mpc(0.0, full_state(model.x_trim + e0))
    np.testing.assert_allclose(u - model.u_trim, mpc.K @ e0, atol=1e-8)


def test_polytope_mpc_from_inside_the_terminal_set_also_equals_lqr(model):
    mpc = make_mpc(model, terminal="polytope")
    e0 = 0.01 * np.array([1.0, -1.0, 1.0, 0.5, 0.0, 0.0, 0.2, 0.0, 0.0])
    u = mpc(0.0, full_state(model.x_trim + e0))
    np.testing.assert_allclose(u - model.u_trim, mpc.K @ e0, atol=1e-7)
    assert mpc.fallback_count == 0


def test_equality_mpc_plans_to_zero_terminal_error(model):
    mpc = make_mpc(model, terminal="equality", horizon=20)
    e0 = np.array([0.3, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mpc(0.0, full_state(model.x_trim + e0))
    e_N = mpc._Phi_N @ e0 + mpc._Gamma_N @ mpc._previous
    np.testing.assert_allclose(e_N, 0.0, atol=1e-6)
    assert mpc.fallback_count == 0


def test_infeasible_equality_falls_back_and_is_logged(model):
    # 3 steps cannot remove a 2 m position error within the rate limits.
    mpc = make_mpc(model, terminal="equality", horizon=3)
    e0 = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u = mpc(0.0, full_state(model.x_trim + e0))
    assert mpc.fallback_count == 1
    assert np.all(u >= mpc.settings.u_lb) and np.all(u <= mpc.settings.u_ub)


def test_setpoint_reference_is_the_trim_at_that_position(model):
    reference = LinearizedReference(model, setpoint([1.0, -2.0, -1.0]))
    dx, du = reference(3.0)
    np.testing.assert_allclose(dx, [1.0, -2.0, -1.0, 0, 0, 0, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(du, 0.0, atol=1e-12)


def test_circle_reference_is_nearly_a_trajectory_of_the_discrete_model(model):
    circle = CircleReference([0.0, 0.0, -2.0], radius=2.0, vertical_amplitude=0.0,
                             period=10.0, yaw=0.0, gravity=9.81)
    reference = LinearizedReference(model, circle.derivatives)
    residuals, steps = [], []
    for k in range(100):
        dx, du = reference(k * DT)
        dx_next, _ = reference((k + 1) * DT)
        residuals.append(np.linalg.norm(model.A @ dx + model.B @ du - dx_next))
        steps.append(np.linalg.norm(dx_next - dx))
    # Continuous-time inversion sampled with a ZOH: the residual is a small
    # fraction of the per-step motion of the reference.
    assert max(residuals) < 0.05 * max(steps)


def test_circle_reference_tilts_toward_the_centre(vehicle):
    # No wind, so drag has no linear term and the tilt is purely centripetal.
    still_air = HoverModel.build(vehicle, np.zeros(3), yaw=0.0, dt=DT)
    circle = CircleReference([0.0, 0.0, -2.0], radius=2.0, vertical_amplitude=0.0,
                             period=10.0, yaw=0.0, gravity=9.81)
    reference = LinearizedReference(still_air, circle.derivatives)
    dx, _ = reference(0.0)   # at t = 0, p = c + (0, −r): the centre is east, a = +r Ω² east
    # East acceleration needs positive roll: a_y = g φ in the hover model.
    assert dx[6] == pytest.approx(circle.radius * circle.omega**2 / 9.81, rel=1e-6)


def test_outer_state_round_trips_euler(model):
    x9 = np.array([1.0, 2.0, 3.0, 0.1, -0.2, 0.3, 0.05, -0.04, 0.3])
    np.testing.assert_allclose(outer_state(full_state(x9)), x9, atol=1e-12)


def test_mpc_holds_hover_in_closed_loop_without_wind(vehicle):
    from controllers.low_level import BodyRateController
    from datagen.trajectories import SetpointReference
    from simulation.outer_loop import simulate_outer_loop
    from wind.wind_model import ConstantWind

    still_air = HoverModel.build(vehicle, np.zeros(3), yaw=0.0, dt=DT)
    mpc = make_mpc(still_air, terminal="polytope", derivatives=setpoint([0.0, 0.0, -1.0]),
                   duration=3.0)
    x0 = full_state(np.array([0.2, -0.1, -0.8, 0, 0, 0, 0, 0, 0]))
    log = simulate_outer_loop(x0, 3.0, SetpointReference([0.0, 0.0, -1.0], 0.0, 9.81),
                              ConstantWind(np.zeros(3)), vehicle, mpc,
                              BodyRateController(vehicle, [20.0, 20.0, 10.0]))
    error = log.x[-1, rigid_body.POS] - np.array([0.0, 0.0, -1.0])
    assert np.linalg.norm(error) < 0.01   # converged within 3 s from a 0.3 m offset
    assert mpc.fallback_count == 0
    np.testing.assert_allclose(log.u_cmd, log.u_h)
