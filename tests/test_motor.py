"""Motor model (dynamics/motor.py) against analytic cases."""
import numpy as np
import pytest

from dynamics.motor import MotorParams, load_motor_params, motor_step, steady_speed

K_Q = 1.18e-7          # N·m·s², configs/rotor.toml
PARAMS = MotorParams(time_constant=0.03, rotor_inertia=3e-5)


def test_hover_with_nominal_torque_is_an_equilibrium():
    omega_cmd = np.full(4, 576.0)
    omega = motor_step(omega_cmd, omega_cmd, K_Q * omega_cmd**2, K_Q, PARAMS, 1e-3)
    np.testing.assert_allclose(omega, omega_cmd, rtol=0, atol=1e-12)


def test_step_response_matches_first_order_exponential():
    # Nominal torque at the command, held: Ω(t) = Ω_cmd + (Ω0 − Ω_cmd) exp(−t/τ_m).
    omega_cmd = np.array([600.0])
    omega0 = np.array([500.0])
    dt, n = 1e-3, 90
    omega = omega0
    for _ in range(n):
        omega = motor_step(omega, omega_cmd, K_Q * omega_cmd**2, K_Q, PARAMS, dt)
    expected = omega_cmd + (omega0 - omega_cmd) * np.exp(-n * dt / PARAMS.time_constant)
    np.testing.assert_allclose(omega, expected, rtol=1e-12)


def test_two_half_steps_equal_one_full_step():
    args = (np.array([550.0]), np.array([580.0]), np.array([0.041]), K_Q, PARAMS)
    full = motor_step(*args, 2e-3)
    half = motor_step(motor_step(*args, 1e-3), *args[1:], 1e-3)
    np.testing.assert_allclose(half, full, rtol=1e-14)


def test_extra_aerodynamic_torque_slows_the_rotor_by_dq_over_km():
    omega_cmd = np.array([576.0])
    extra = 0.02 * K_Q * omega_cmd**2          # +2 % torque, as BEMT in edgewise flow
    omega_ss = steady_speed(omega_cmd, K_Q * omega_cmd**2 + extra, K_Q, PARAMS)
    np.testing.assert_allclose(omega_cmd - omega_ss, extra / PARAMS.stiffness, rtol=1e-12)
    assert np.all(omega_ss < omega_cmd)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        MotorParams(time_constant=0.0, rotor_inertia=3e-5)
    with pytest.raises(ValueError):
        motor_step(np.ones(2), np.ones(3), np.ones(2), K_Q, PARAMS, 1e-3)
    with pytest.raises(ValueError):
        motor_step(np.array([-1.0]), np.ones(1), np.ones(1), K_Q, PARAMS, 1e-3)
    with pytest.raises(ValueError):
        motor_step(np.ones(1), np.ones(1), np.array([np.nan]), K_Q, PARAMS, 1e-3)


def test_load_motor_params():
    assert load_motor_params(None) is None
    assert load_motor_params({"model": "ideal"}) is None
    params = load_motor_params({"model": "first_order", "time_constant": 0.03,
                                "rotor_inertia": 3e-5})
    assert params.stiffness == pytest.approx(1e-3)
    with pytest.raises(ValueError):
        load_motor_params({"model": "first_order", "time_constant": 0.03})
    with pytest.raises(ValueError):
        load_motor_params({"model": "esc"})
