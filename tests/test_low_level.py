"""Body-rate controller against the rigid-body model (NED/FRD)."""
from dataclasses import replace

import numpy as np
import pytest

from controllers.low_level import BodyRateController
from dynamics.dynamics import QUAT, RATE, STATE_DIM, VEL, rk4_step, state_derivative
from dynamics.quaternion import Quaternion

GAINS = np.array([20.0, 20.0, 10.0])
ZERO = np.zeros(3)


def make_state(q=Quaternion.identity().to_numpy(), omega=ZERO) -> np.ndarray:
    x = np.zeros(STATE_DIM)
    x[QUAT], x[RATE] = q, omega
    return x


def test_hover_command_is_equilibrium(params):
    controller = BodyRateController(params, GAINS)
    x = make_state()
    f_t, tau_t = controller(x, [0.0, 0.0, 0.0, params.gravity])
    np.testing.assert_allclose(state_derivative(x, f_t, tau_t, ZERO, params), 0.0, atol=1e-12)


def test_tracked_rate_is_held_against_gyroscopic_coupling(params):
    # ω = ω_des about a non-principal axis: without the ω × Jω term the
    # asymmetric inertia in the fixture would make ω̇ non-zero.
    controller = BodyRateController(params, GAINS)
    omega = np.array([1.0, -2.0, 0.5])
    x = make_state(omega=omega)
    f_t, tau_t = controller(x, [*omega, params.gravity])
    np.testing.assert_allclose(state_derivative(x, f_t, tau_t, ZERO, params)[RATE], 0.0, atol=1e-12)


def test_rate_step_is_first_order_with_gain_time_constant(params):
    # Closed loop ω̇ = K_ω (ω_des − ω) per axis, even with cross-axis
    # gyroscopic coupling. The torque is held over each step (zero-order hold),
    # so the error shrinks by (1 − k dt) per step, not e^{−k dt}; the two
    # differ by ~1e-3 here. 1 kHz loop, drag off.
    no_drag = replace(params, drag_coeffs=ZERO)
    controller = BodyRateController(no_drag, GAINS)
    omega_des = np.array([1.0, -0.5, 0.8])
    u_h = [*omega_des, no_drag.gravity]
    dt, n_steps = 1e-3, 200
    x = make_state()
    for _ in range(n_steps):
        f_t, tau_t = controller(x, u_h)
        x = rk4_step(x, f_t, tau_t, ZERO, no_drag, dt)

    expected = omega_des * (1.0 - (1.0 - GAINS * dt) ** n_steps)
    np.testing.assert_allclose(x[RATE], expected, atol=1e-4)


def test_thrust_points_along_negative_body_z(params):
    controller = BodyRateController(params, GAINS)
    f_t, _ = controller(make_state(), [0.0, 0.0, 0.0, 12.0])
    np.testing.assert_allclose(f_t, [0.0, 0.0, -params.mass * 12.0])


def test_tilted_hover_thrust_accelerates_sideways(params):
    # 30° roll (right side down in FRD) tilts the thrust axis −z_body toward
    # +y NED (east), so the vehicle accelerates to its right.
    no_drag = replace(params, drag_coeffs=ZERO)
    controller = BodyRateController(no_drag, GAINS)
    x = make_state(q=Quaternion.from_euler([np.radians(30.0), 0.0, 0.0]).to_numpy())
    f_t, tau_t = controller(x, [0.0, 0.0, 0.0, no_drag.gravity])
    v_dot = state_derivative(x, f_t, tau_t, ZERO, no_drag)[VEL]
    assert v_dot[1] > 0


@pytest.mark.parametrize("gains", [[20.0, 20.0, 0.0], [20.0, -1.0, 10.0], [20.0, 20.0]])
def test_invalid_gains_raise(params, gains):
    with pytest.raises(ValueError):
        BodyRateController(params, gains)


def test_invalid_command_raises(params):
    with pytest.raises(ValueError):
        BodyRateController(params, GAINS)(make_state(), [0.0, np.nan, 0.0, 9.81])
