"""Hover linearization: trim, Jacobians against the hand derivation, ZOH closed forms."""
import numpy as np
import pytest

from controllers.hover_model import (
    EULER, HoverModel, PITCH, POS, ROLL, THRUST, VEL, YAW, discretize, euler_dynamics,
    hover_trim, linearize, rotation_zyx,
)
from dynamics.params import QuadParams
from dynamics.quaternion import Quaternion


@pytest.fixture
def isotropic_params() -> QuadParams:
    """1 kg vehicle with the drag of configs/quadrotor.toml."""
    return QuadParams(mass=1.0, inertia=np.diag([0.0082, 0.0082, 0.0149]),
                      drag_coeffs=np.array([0.05, 0.05, 0.10]), gravity=9.81)


def test_rotation_matches_quaternion_convention():
    euler = np.array([0.2, -0.3, 1.1])
    R_q = Quaternion.from_euler(euler).to_rotation_matrix()
    np.testing.assert_allclose(rotation_zyx(euler), R_q, atol=1e-12)


def test_no_wind_trim_is_level_hover(params):
    x_star, u_star = hover_trim(np.zeros(3), yaw=0.4, params=params)
    np.testing.assert_allclose(x_star, [0, 0, 0, 0, 0, 0, 0, 0, 0.4], atol=1e-12)
    np.testing.assert_allclose(u_star, [0, 0, 0, params.gravity], atol=1e-12)


def test_trim_in_wind_is_an_equilibrium_and_leans_into_the_wind(isotropic_params):
    wind = np.array([2.7032, 1.7566, 0.0124])
    x_star, u_star = hover_trim(wind, yaw=0.0, params=isotropic_params)
    residual = euler_dynamics(x_star, u_star, wind, isotropic_params)
    assert np.max(np.abs(residual)) < 1e-12
    # Wind toward +x (north) needs nose-up pitch; toward +y (east) needs negative roll.
    assert x_star[PITCH] == pytest.approx(0.0444, abs=5e-4)
    assert x_star[ROLL] == pytest.approx(-0.0288, abs=5e-4)
    assert u_star[THRUST] == pytest.approx(9.855, abs=1e-3)


def test_no_wind_jacobians_match_hand_derivation(params):
    g = params.gravity
    x_star, u_star = hover_trim(np.zeros(3), yaw=0.0, params=params)
    A_c, B_c, D_c = linearize(x_star, u_star, np.zeros(3), params)
    A_expected = np.zeros((9, 9))
    A_expected[POS, VEL] = np.eye(3)
    A_expected[3, PITCH] = -g   # positive pitch accelerates toward −x (south)
    A_expected[4, ROLL] = g     # positive roll accelerates toward +y (east)
    B_expected = np.zeros((9, 4))
    B_expected[5, THRUST] = -1.0
    B_expected[EULER, 0:3] = np.eye(3)
    # At zero airspeed the central difference of v‖v‖ returns the step h itself,
    # so drag entries are O(D h / m) ≈ 2e-7 rather than 0.
    np.testing.assert_allclose(A_c, A_expected, atol=1e-6)
    np.testing.assert_allclose(B_c, B_expected, atol=1e-6)
    np.testing.assert_allclose(D_c, 0.0, atol=1e-6)  # quadratic drag has no slope at rest


def test_yaw_rotates_which_tilt_drives_which_axis(params):
    g = params.gravity
    x_star, u_star = hover_trim(np.zeros(3), yaw=np.pi / 2, params=params)
    A_c, _, _ = linearize(x_star, u_star, np.zeros(3), params)
    # Nose east: pitch now drives east (−y), roll drives north (−x).
    np.testing.assert_allclose(A_c[VEL, ROLL], [-g, 0, 0], atol=1e-6)
    np.testing.assert_allclose(A_c[VEL, PITCH], [0, -g, 0], atol=1e-6)


def test_drag_damps_velocity_and_wind_enters_with_opposite_sign(isotropic_params):
    wind = np.array([2.7032, 1.7566, 0.0124])
    x_star, u_star = hover_trim(wind, yaw=0.0, params=isotropic_params)
    A_c, _, D_c = linearize(x_star, u_star, wind, isotropic_params)
    damping = A_c[VEL, VEL]
    assert np.all(np.linalg.eigvals(damping).real < 0)
    np.testing.assert_allclose(D_c[VEL], -damping, atol=1e-7)
    # Quadratic drag: about 2 D ‖v̄‖ along the wind and D ‖v̄‖ across it.
    speed = np.linalg.norm(wind)
    rates = np.sort(-np.linalg.eigvals(damping).real)
    np.testing.assert_allclose(rates, [0.05 * speed, 0.1 * speed, 0.1 * speed], rtol=0.1)


def test_zoh_matches_closed_form_integrator_chains(params):
    g, dt = params.gravity, 0.1
    x_star, u_star = hover_trim(np.zeros(3), yaw=0.0, params=params)
    A, B, _ = discretize(*linearize(x_star, u_star, np.zeros(3), params), dt)
    # North chain [p_x, v_x, θ] driven by ω_y: nilpotent, so the series stops at dt².
    idx = [0, 3, PITCH]
    np.testing.assert_allclose(A[np.ix_(idx, idx)],
                               [[1, dt, -g * dt**2 / 2], [0, 1, -g * dt], [0, 0, 1]], atol=1e-9)
    np.testing.assert_allclose(B[idx, 1], [-g * dt**3 / 6, -g * dt**2 / 2, dt], atol=1e-9)
    # Down chain [p_z, v_z] driven by c; yaw is a single integrator of ω_z.
    np.testing.assert_allclose(B[[2, 5], THRUST], [-dt**2 / 2, -dt], atol=1e-9)
    assert B[YAW, 2] == pytest.approx(dt)


def test_hover_model_is_controllable(isotropic_params):
    model = HoverModel.build(isotropic_params, [2.7032, 1.7566, 0.0124], 0.0, 0.1)
    ctrb = np.hstack([np.linalg.matrix_power(model.A, k) @ model.B for k in range(9)])
    assert np.linalg.matrix_rank(ctrb) == 9


def test_excessive_trim_tilt_is_rejected(isotropic_params):
    with pytest.raises(ValueError, match="tilt"):
        hover_trim(np.array([40.0, 0.0, 0.0]), 0.0, isotropic_params)
