"""Rigid-body dynamics against analytic cases (NED, z down)."""
from dataclasses import replace

import numpy as np
import pytest

from dynamics.dynamics import (
    POS,
    QUAT,
    RATE,
    STATE_DIM,
    VEL,
    rk4_step,
    state_derivative,
)
from dynamics.params import QuadParams, load_params
from dynamics.quaternion import Quaternion
from conftest import REPO_ROOT

ZERO = np.zeros(3)
IDENTITY = Quaternion.identity().to_numpy()


def make_state(p=ZERO, q=IDENTITY, v=ZERO, omega=ZERO) -> np.ndarray:
    x = np.empty(STATE_DIM)
    x[POS], x[QUAT], x[VEL], x[RATE] = p, q, v, omega
    return x


def hover_thrust(params: QuadParams) -> np.ndarray:
    return np.array([0.0, 0.0, -params.mass * params.gravity])


def simulate(x, f_t, tau_t, v_w, params, dt, n_steps):
    for _ in range(n_steps):
        x = rk4_step(x, f_t, tau_t, v_w, params, dt)
    return x


def test_hover_is_equilibrium(params):
    x_dot = state_derivative(make_state(), hover_thrust(params), ZERO, ZERO, params)
    np.testing.assert_allclose(x_dot, np.zeros(STATE_DIM), atol=1e-12)


def test_hover_drifting_with_steady_wind_is_equilibrium(params):
    # Moving with the air mass means zero air-relative velocity, hence no drag.
    wind = np.array([3.0, -2.0, 0.5])
    x_dot = state_derivative(make_state(v=wind), hover_thrust(params), ZERO, wind, params)
    np.testing.assert_allclose(x_dot[VEL], ZERO, atol=1e-12)


def test_free_fall_accelerates_down_at_g(params):
    # Drag off: with drag the fall is slower than ½gt², so this case isolates gravity.
    no_drag = replace(params, drag_coeffs=ZERO)
    t, dt = 1.0, 0.01
    x = simulate(make_state(), ZERO, ZERO, ZERO, no_drag, dt, round(t / dt))
    # Quadratic in t, so RK4 is exact up to round-off.
    np.testing.assert_allclose(x[POS], [0, 0, 0.5 * params.gravity * t**2], atol=1e-10)
    np.testing.assert_allclose(x[VEL], [0, 0, params.gravity * t], atol=1e-10)


def test_drag_opposes_air_relative_velocity(params):
    v = np.array([3.0, 0.0, 0.0])
    x_dot = state_derivative(make_state(v=v), hover_thrust(params), ZERO, ZERO, params)
    expected_ax = -params.drag_coeffs[0] * 3.0 * 3.0 / params.mass
    np.testing.assert_allclose(x_dot[VEL], [expected_ax, 0, 0], atol=1e-12)


def test_wind_pushes_stationary_vehicle_downwind(params):
    wind = np.array([0.0, 4.0, 0.0])
    x_dot = state_derivative(make_state(), hover_thrust(params), ZERO, wind, params)
    expected_ay = params.drag_coeffs[1] * 4.0 * 4.0 / params.mass
    np.testing.assert_allclose(x_dot[VEL], [0, expected_ay, 0], atol=1e-12)


def test_constant_yaw_rate_integrates_to_expected_heading(params):
    rate, t, dt = 0.5, 2.0, 0.01
    x = make_state(omega=[0.0, 0.0, rate])
    x = simulate(x, hover_thrust(params), ZERO, ZERO, params, dt, round(t / dt))
    np.testing.assert_allclose(Quaternion(x[QUAT]).to_euler(), [0, 0, rate * t], atol=1e-9)


def test_torque_free_spin_conserves_momentum_and_energy(params):
    # Gravity and thrust do not affect attitude; use free fall to isolate rotation.
    J = params.inertia
    x0 = make_state(omega=[1.0, 2.0, 0.5])

    def world_momentum(x):
        return Quaternion(x[QUAT]).to_rotation_matrix() @ J @ x[RATE]

    def kinetic_energy(x):
        return 0.5 * x[RATE] @ J @ x[RATE]

    x = simulate(x0, ZERO, ZERO, ZERO, params, 0.001, 3000)
    np.testing.assert_allclose(world_momentum(x), world_momentum(x0), rtol=1e-6)
    np.testing.assert_allclose(kinetic_energy(x), kinetic_energy(x0), rtol=1e-6)
    assert np.isclose(np.linalg.norm(x[QUAT]), 1.0)


def test_rk4_rejects_bad_inputs(params):
    x = make_state()
    with pytest.raises(ValueError, match="dt"):
        rk4_step(x, ZERO, ZERO, ZERO, params, 0.0)
    with pytest.raises(ValueError, match="shape"):
        rk4_step(x[:12], ZERO, ZERO, ZERO, params, 0.01)
    with pytest.raises(ValueError, match="NaN"):
        rk4_step(x, [0, 0, np.nan], ZERO, ZERO, params, 0.01)
    bad_q = make_state(q=[1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="unit norm"):
        state_derivative(bad_q, ZERO, ZERO, ZERO, params)


def test_params_reject_non_positive_definite_inertia():
    with pytest.raises(ValueError, match="positive definite"):
        QuadParams(mass=1.0, inertia=np.diag([0.01, 0.0, 0.02]),
                   drag_coeffs=ZERO, gravity=9.81)


def test_params_are_read_only(params):
    with pytest.raises(ValueError):
        params.inertia[0, 0] = 1.0


def test_repo_config_loads():
    p = load_params(REPO_ROOT / "configs" / "quadrotor.toml")
    np.testing.assert_allclose(p.gravity_vector, [0, 0, 9.81])
