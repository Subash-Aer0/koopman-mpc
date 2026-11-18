"""Linearization along a trajectory: the reference is a solution of the nonlinear model."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from controllers.hover_model import (
    EULER, HoverModel, PITCH, ROLL, THRUST, euler_dynamics, hover_trim, rotation_zyx,
)
from controllers.trajectory_model import LTVModel, trajectory_point
from datagen.trajectories import CircleReference, flat_state
from dynamics.params import QuadParams
from dynamics.quaternion import Quaternion

WIND = np.array([2.7032, 1.7566, 0.0124])   # mean wind of configs/controllers/hover_mpc.toml
DT = 0.1


@pytest.fixture
def isotropic_params() -> QuadParams:
    """1 kg vehicle with the drag of configs/quadrotor.toml."""
    return QuadParams(mass=1.0, inertia=np.diag([0.0082, 0.0082, 0.0149]),
                      drag_coeffs=np.array([0.05, 0.05, 0.10]), gravity=9.81)


@pytest.fixture
def dragless_params() -> QuadParams:
    """No drag, so the reference reduces to the flatness of sdlqr_model (no drag, no wind)."""
    return QuadParams(mass=1.0, inertia=np.diag([0.0082, 0.0082, 0.0149]),
                      drag_coeffs=np.zeros(3), gravity=9.81)


def _circle(period=5.0, radius=2.0, height=0.0, yaw=0.0, gravity=9.81):
    return CircleReference([0.0, 0.0, -2.0], radius, height, period, yaw, gravity)


def _point(reference, t, wind, params):
    return trajectory_point(*reference.derivatives(t), reference.yaw, wind, params)


@pytest.mark.parametrize("wind", [np.zeros(3), WIND])
def test_reference_at_rest_is_the_hover_trim(isotropic_params, wind):
    p = np.array([1.0, -2.0, -3.0])
    zero = np.zeros(3)
    x_ref, u_ref = trajectory_point(p, zero, zero, zero, 0.4, wind, isotropic_params)
    x_star, u_star = hover_trim(wind, 0.4, isotropic_params)
    np.testing.assert_allclose(x_ref[3:], x_star[3:], atol=1e-12)
    np.testing.assert_allclose(x_ref[:3], p)
    np.testing.assert_allclose(u_ref, u_star, atol=1e-12)


def test_dragless_circle_tilt_is_the_exact_not_the_linearized_one(dragless_params):
    g = dragless_params.gravity
    circle = _circle(period=5.0, radius=2.0)
    centripetal = circle.radius * circle.omega**2
    for t in np.linspace(0.0, 5.0, 7):
        x_ref, u_ref = _point(circle, t, np.zeros(3), dragless_params)
        tilt = np.arccos(np.cos(x_ref[ROLL]) * np.cos(x_ref[PITCH]))
        # The hover model would give centripetal / g = 0.322 rad.
        assert tilt == pytest.approx(np.arctan(centripetal / g), abs=1e-10)
        assert u_ref[THRUST] == pytest.approx(np.hypot(g, centripetal), abs=1e-10)


def test_dragless_thrust_matches_flatness_of_the_lqr_model(dragless_params):
    # Only the yaw-independent parts are compared: flat_state fixes the heading by
    # b₂ ⟂ x_ψ, this module by a constant ZYX Euler yaw. They agree on c and b₃ = R e₃.
    circle = _circle(period=6.0, radius=2.5, height=0.5, yaw=0.3)
    for t in np.linspace(0.0, 6.0, 9):
        x_ref, u_ref = _point(circle, t, np.zeros(3), dragless_params)
        x_flat, u_flat = flat_state(*circle.derivatives(t), circle.yaw, circle.gravity)
        R_flat = Quaternion(x_flat[3:7]).to_rotation_matrix()
        np.testing.assert_allclose(rotation_zyx(x_ref[EULER])[:, 2], R_flat[:, 2], atol=1e-10)
        assert u_ref[THRUST] == pytest.approx(u_flat[3], abs=1e-10)


@pytest.mark.parametrize("wind", [np.zeros(3), WIND])
def test_reference_satisfies_the_nonlinear_model(isotropic_params, wind):
    # ẋ_ref by a central difference in time must equal euler_dynamics(x_ref, u_ref):
    # the velocity rows check the drag-aware attitude and thrust, the angle rows
    # check the rates from the implicit-function derivative.
    circle = _circle(period=5.0, radius=2.0, height=0.5, yaw=0.3)
    h = 1e-4   # s; truncation ~h² ≈ 1e-8, round-off ~1e-13 / h ≈ 1e-9
    for t in np.linspace(0.0, 5.0, 11):
        x_ref, u_ref = _point(circle, t, wind, isotropic_params)
        x_plus, _ = _point(circle, t + h, wind, isotropic_params)
        x_minus, _ = _point(circle, t - h, wind, isotropic_params)
        x_dot = (x_plus - x_minus) / (2 * h)
        np.testing.assert_allclose(euler_dynamics(x_ref, u_ref, wind, isotropic_params),
                                   x_dot, atol=1e-6)


def test_ltv_model_on_a_setpoint_is_the_hover_model(isotropic_params):
    p = np.array([1.0, -2.0, -3.0])
    zero = np.zeros(3)
    ltv = LTVModel.build(lambda t: (p, zero, zero, zero), 1.0, isotropic_params, WIND, 0.0, DT)
    hover = HoverModel.build(isotropic_params, WIND, 0.0, DT)
    assert len(ltv.t) == 11
    for k in range(len(ltv.t)):
        np.testing.assert_allclose(ltv.A[k], hover.A, atol=1e-9)
        np.testing.assert_allclose(ltv.B[k], hover.B, atol=1e-9)
        np.testing.assert_allclose(ltv.D[k], hover.D, atol=1e-9)
        np.testing.assert_allclose(ltv.x_ref[k], hover.x_trim + np.r_[p, zero, zero], atol=1e-12)
        np.testing.assert_allclose(ltv.u_ref[k], hover.u_trim, atol=1e-12)
    np.testing.assert_allclose(ltv.r, 0.0, atol=1e-12)   # an equilibrium is a discrete trajectory


def _nonlinear_step(x, u, wind, params):
    """Independent integrator (adaptive RK45) for one ZOH sample."""
    sol = solve_ivp(lambda _, s: euler_dynamics(s, u, wind, params), (0.0, DT), x,
                    rtol=1e-12, atol=1e-12)
    return sol.y[:, -1]


def test_prediction_is_exact_on_the_reference_and_first_order_off_it(isotropic_params):
    circle = _circle(period=5.0, radius=2.0, height=0.5, yaw=0.3)
    ltv = LTVModel.build(circle.derivatives, 2.0, isotropic_params, WIND, circle.yaw, DT)
    rng = np.random.default_rng(0)
    direction_e = rng.standard_normal(9)
    direction_u = rng.standard_normal(4)
    for k in (0, 7, 15):
        x_k, u_k, x_next = ltv.x_ref[k], ltv.u_ref[k], ltv.x_ref[k + 1]
        # On the reference, x_ref,k+1 + r_k is the nonlinear step.
        np.testing.assert_allclose(x_next + ltv.r[k],
                                   _nonlinear_step(x_k, u_k, WIND, isotropic_params), atol=1e-9)
        # Off it, the linear prediction error is second order in the perturbation.
        # Regression: expm of the Jacobians frozen at t_k gave a ratio of 2.3 here
        # (a first-order error), because the attitude moves within the sample.
        errors = []
        for scale in (1e-2, 5e-3):
            e, du = scale * direction_e, scale * direction_u
            actual = _nonlinear_step(x_k + e, u_k + du, WIND, isotropic_params) - x_next
            predicted = ltv.A[k] @ e + ltv.B[k] @ du + ltv.r[k]
            errors.append(np.linalg.norm(actual - predicted))
        assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.15)


def test_residual_is_small_compared_with_the_motion_per_sample(isotropic_params):
    circle = _circle(period=5.0, radius=2.0)
    ltv = LTVModel.build(circle.derivatives, 5.0, isotropic_params, WIND, circle.yaw, DT)
    motion = np.linalg.norm(np.diff(ltv.x_ref, axis=0), axis=1)
    assert np.max(np.linalg.norm(ltv.r, axis=1) / motion) < 0.05


def test_free_fall_and_bad_inputs_are_rejected(isotropic_params):
    zero = np.zeros(3)
    g = isotropic_params.gravity
    with pytest.raises(ValueError, match="free fall"):
        trajectory_point(zero, zero, np.array([0.0, 0.0, g]), zero, 0.0, zero, isotropic_params)
    with pytest.raises(ValueError, match="NaN"):
        trajectory_point(zero, np.array([np.nan, 0, 0]), zero, zero, 0.0, zero, isotropic_params)
    with pytest.raises(ValueError, match="dt"):
        LTVModel.build(lambda t: (zero, zero, zero, zero), 1.0, isotropic_params, zero, 0.0, 0.0)
