"""SD-LQR nominal model and Jacobians against analytic cases and finite differences."""
from dataclasses import replace

import numpy as np
import pytest

from controllers.sdlqr_model import (
    INPUT_DIM,
    POS,
    QUAT,
    STATE_DIM,
    THRUST,
    VEL,
    discretize,
    linearize,
    nominal_dynamics,
    tangent_basis,
)
from dynamics import dynamics as rigid_body
from dynamics.quaternion import Quaternion

G = 9.81
IDENTITY = Quaternion.identity().to_numpy()
HOVER_U = np.array([0.0, 0.0, 0.0, G])


def make_state(p=np.zeros(3), q=IDENTITY, v=np.zeros(3)) -> np.ndarray:
    x = np.empty(STATE_DIM)
    x[POS], x[QUAT], x[VEL] = p, q, v
    return x


def numerical_jacobians(x, u, eps=1e-6):
    """Central differences of nominal_dynamics; error O(eps²)."""
    A = np.empty((STATE_DIM, STATE_DIM))
    B = np.empty((STATE_DIM, INPUT_DIM))
    for i in range(STATE_DIM):
        dx = np.zeros(STATE_DIM)
        dx[i] = eps
        A[:, i] = (nominal_dynamics(x + dx, u, G) - nominal_dynamics(x - dx, u, G)) / (2 * eps)
    for j in range(INPUT_DIM):
        du = np.zeros(INPUT_DIM)
        du[j] = eps
        B[:, j] = (nominal_dynamics(x, u + du, G) - nominal_dynamics(x, u - du, G)) / (2 * eps)
    return A, B


TILTED = make_state(
    p=[1.0, -2.0, -3.0],
    q=Quaternion.from_euler([0.3, -0.2, 1.1]).to_numpy(),
    v=[0.5, -1.0, 0.2],
)
MANEUVER_U = np.array([0.4, -0.7, 0.9, 12.0])


def test_hover_is_equilibrium():
    np.testing.assert_allclose(nominal_dynamics(make_state(), HOVER_U, G), 0.0, atol=1e-12)


def test_matches_full_rigid_body_without_drag_or_wind(params):
    # Guards the NED/FRD sign convention: c along −z_body must equal f_t = [0, 0, −m c].
    no_drag = replace(params, drag_coeffs=np.zeros(3))
    x13 = np.concatenate([TILTED, MANEUVER_U[:3]])
    f_t = np.array([0.0, 0.0, -no_drag.mass * MANEUVER_U[THRUST]])
    x13_dot = rigid_body.state_derivative(x13, f_t, np.zeros(3), np.zeros(3), no_drag)
    np.testing.assert_allclose(
        nominal_dynamics(TILTED, MANEUVER_U, no_drag.gravity), x13_dot[:STATE_DIM], atol=1e-12
    )


def test_hover_thrust_increase_accelerates_up():
    _, B = linearize(make_state(), HOVER_U)
    np.testing.assert_allclose(B[VEL, THRUST], [0.0, 0.0, -1.0])


@pytest.mark.parametrize(
    "x, u",
    [
        (make_state(), HOVER_U),
        (TILTED, MANEUVER_U),
        # Off the unit sphere: exercises the normalization Jacobian, Eq. (18).
        (make_state(q=1.3 * Quaternion.from_euler([-0.5, 0.4, -2.0]).to_numpy()), MANEUVER_U),
    ],
    ids=["hover", "tilted", "non_unit_quaternion"],
)
def test_jacobians_match_finite_differences(x, u):
    A, B = linearize(x, u)
    A_num, B_num = numerical_jacobians(x, u)
    np.testing.assert_allclose(A, A_num, atol=1e-8)
    np.testing.assert_allclose(B, B_num, atol=1e-8)


def test_quaternion_norm_direction_is_in_null_space_of_A():
    # Scaling q does not change f, so A has a zero eigenvalue along q.
    A, _ = linearize(TILTED, MANEUVER_U)
    direction = np.zeros(STATE_DIM)
    direction[QUAT] = TILTED[QUAT]
    np.testing.assert_allclose(A @ direction, 0.0, atol=1e-12)


def test_tangent_basis_is_orthonormal_and_orthogonal_to_norm_direction():
    q = TILTED[QUAT]
    T = tangent_basis(q)
    norm_direction = np.zeros(STATE_DIM)
    norm_direction[QUAT] = q
    np.testing.assert_allclose(T.T @ T, np.eye(STATE_DIM - 1), atol=1e-12)
    np.testing.assert_allclose(T.T @ norm_direction, 0.0, atol=1e-12)


def test_tangent_basis_spans_input_directions():
    # B's range lies in span(T), so projecting onto T loses no control authority.
    _, B = linearize(TILTED, MANEUVER_U)
    T = tangent_basis(TILTED[QUAT])
    np.testing.assert_allclose(T @ (T.T @ B), B, atol=1e-12)


def test_tangent_basis_rejects_zero_quaternion():
    with pytest.raises(ValueError):
        tangent_basis(np.zeros(4))


def test_discretize_double_integrator():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    dt = 0.1
    A_d, B_d = discretize(A, B, dt)
    np.testing.assert_allclose(A_d, [[1.0, dt], [0.0, 1.0]], atol=1e-14)
    np.testing.assert_allclose(B_d, [[0.5 * dt**2], [dt]], atol=1e-14)


def test_discretize_rejects_bad_dt():
    with pytest.raises(ValueError):
        discretize(np.zeros((2, 2)), np.zeros((2, 1)), 0.0)


@pytest.mark.parametrize(
    "x, u",
    [
        (np.zeros(STATE_DIM), HOVER_U),
        (make_state()[:9], HOVER_U),
        (make_state(), HOVER_U[:3]),
        (make_state(), np.array([0.0, np.nan, 0.0, G])),
    ],
    ids=["zero_quaternion", "short_state", "short_input", "nan_input"],
)
def test_invalid_inputs_raise(x, u):
    with pytest.raises(ValueError):
        linearize(x, u)
