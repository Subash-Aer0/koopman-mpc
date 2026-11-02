"""Nominal model and linearization for the state-dependent LQR (Foehn & Scaramuzza [16]).

The LQR controls only the slow dynamics; body rates are an input tracked by
the low-level controller (Sec. II-A of [16]). The model has no drag and no
wind: the baseline is designed without wind knowledge.

State x = [p (3), q (4), v (3)], Eq. (1) of [16]:
    p  position, NED world frame, m
    q  quaternion, body FRD -> world NED, scalar first (see quaternion.py)
    v  inertial velocity, NED world frame, m/s
Input u = [ω (3), c], Eq. (2) of [16]:
    ω  desired body rates, body FRD frame, rad/s
    c  mass-normalized collective thrust, m/s², c >= 0. Thrust acts along
       −z_body (FRD), i.e. f_t = [0, 0, −m c] in dynamics.py.

Equations, Eq. (3) of [16] rewritten in NED (z down, gravity +g along z):
    ṗ = v
    q̇ = ½ q ⊗ [0, ω]
    v̇ = g e₃ − c R(q) e₃
[16] is z-up, so v̇ there has +c and −g; every v̇ Jacobian below flips sign
relative to Eqs. (22)-(23).

q enters through q_u = q / ‖q‖ (Eq. (18)), so the Jacobians are valid off the
unit sphere and ∂f/∂q has q in its null space: the quaternion norm direction
is neither driven nor penalized by the linear model.
"""
import numpy as np
from scipy.linalg import expm

from dynamics._validation import as_finite_vector

STATE_DIM = 10
INPUT_DIM = 4
POS = slice(0, 3)
QUAT = slice(3, 7)
VEL = slice(7, 10)
RATE = slice(0, 3)
THRUST = 3

_ZERO_NORM_TOL = 1e-12


def _validate(x, u) -> tuple[np.ndarray, np.ndarray]:
    x = as_finite_vector("x", x, STATE_DIM)
    u = as_finite_vector("u", u, INPUT_DIM)
    if np.linalg.norm(x[QUAT]) < _ZERO_NORM_TOL:
        raise ValueError("quaternion in x must be non-zero")
    return x, u


def _normalization_jacobian(q: np.ndarray) -> np.ndarray:
    """∂q_u/∂q = (I − q qᵀ / ‖q‖²) / ‖q‖, Eq. (18) of [16]."""
    norm = np.linalg.norm(q)
    return (np.eye(4) - np.outer(q, q) / norm**2) / norm


def _rate_matrix(omega: np.ndarray) -> np.ndarray:
    """Ω(ω) with q ⊗ [0, ω] = Ω(ω) q, the matrix in Eq. (20) of [16]."""
    wx, wy, wz = omega
    return np.array([
        [0.0, -wx, -wy, -wz],
        [wx, 0.0, wz, -wy],
        [wy, -wz, 0.0, wx],
        [wz, wy, -wx, 0.0],
    ])


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    """G(q) with q ⊗ [0, ω] = G(q) ω, from Eq. (11) of [16]."""
    qw, qx, qy, qz = q
    return np.array([
        [-qx, -qy, -qz],
        [qw, -qz, qy],
        [qz, qw, -qx],
        [-qy, qx, qw],
    ])


def _body_z_axis(q: np.ndarray) -> np.ndarray:
    """R(q) e₃, the body z axis in the world frame (third column of R)."""
    qw, qx, qy, qz = q
    return np.array([
        2.0 * (qx * qz + qw * qy),
        2.0 * (qy * qz - qw * qx),
        qw**2 - qx**2 - qy**2 + qz**2,
    ])


def _body_z_axis_jacobian(q: np.ndarray) -> np.ndarray:
    """∂(R(q) e₃)/∂q, the matrix in Eq. (22) of [16] times 2."""
    qw, qx, qy, qz = q
    return 2.0 * np.array([
        [qy, qz, qw, qx],
        [-qx, -qw, qz, qy],
        [qw, -qx, -qy, qz],
    ])


def nominal_dynamics(x, u, gravity: float) -> np.ndarray:
    """Continuous-time nominal model ẋ = f(x, u), no drag or wind.

    Args:
        x: State [p, q, v], shape (10,); layout and frames in the module docstring.
        u: Input [ω, c], shape (4,); rad/s and m/s².
        gravity: Gravitational acceleration g, m/s², along +z NED.

    Returns:
        ẋ, shape (10,).
    """
    x, u = _validate(x, u)
    q_u = x[QUAT] / np.linalg.norm(x[QUAT])

    x_dot = np.empty(STATE_DIM)
    x_dot[POS] = x[VEL]
    x_dot[QUAT] = 0.5 * _rate_matrix(u[RATE]) @ q_u
    x_dot[VEL] = np.array([0.0, 0.0, gravity]) - u[THRUST] * _body_z_axis(q_u)
    return x_dot


def linearize(x, u) -> tuple[np.ndarray, np.ndarray]:
    """Jacobians A = ∂f/∂x and B = ∂f/∂u at (x, u), Eqs. (16)-(23) of [16] in NED.

    Gravity is constant, so it does not appear in A or B.

    Args:
        x: Linearization state [p, q, v], shape (10,). q need not be unit.
        u: Linearization input [ω, c], shape (4,).

    Returns:
        A, shape (10, 10), and B, shape (10, 4).
    """
    x, u = _validate(x, u)
    q = x[QUAT]
    q_u = q / np.linalg.norm(q)
    dqu_dq = _normalization_jacobian(q)

    A = np.zeros((STATE_DIM, STATE_DIM))
    A[POS, VEL] = np.eye(3)                                               # (19)
    A[QUAT, QUAT] = 0.5 * _rate_matrix(u[RATE]) @ dqu_dq                  # (20)
    A[VEL, QUAT] = -u[THRUST] * _body_z_axis_jacobian(q_u) @ dqu_dq       # (22)

    B = np.zeros((STATE_DIM, INPUT_DIM))
    B[QUAT, RATE] = 0.5 * _quat_matrix(q_u)                               # (21)
    B[VEL, THRUST] = -_body_z_axis(q_u)                                   # (23)
    return A, B


def tangent_basis(q) -> np.ndarray:
    """Orthonormal basis T(q) of state deviations that keep ‖q‖ fixed.

    T = blkdiag(I₃, G(q_u), I₃), shape (10, 9), with G(q_u) the 4×3 matrix of
    _quat_matrix: its columns q_u ⊗ [0, eᵢ] span the tangent space of the unit
    sphere at q_u, GᵀG = I₃ and G Gᵀ = I₄ − q_u q_uᵀ. Since A q = 0 and Bᵀq = 0,
    the reduced pair (Tᵀ A T, Tᵀ B) drops only the uncontrollable norm mode,
    which makes the discrete Riccati equation well posed.

    Args:
        q: Quaternion, scalar first, shape (4,). Normalized internally.

    Returns:
        T, shape (10, 9).
    """
    q = as_finite_vector("q", q, 4)
    norm = np.linalg.norm(q)
    if norm < _ZERO_NORM_TOL:
        raise ValueError("q must be non-zero")

    T = np.zeros((STATE_DIM, STATE_DIM - 1))
    T[POS, 0:3] = np.eye(3)
    T[QUAT, 3:6] = _quat_matrix(q / norm)
    T[VEL, 6:9] = np.eye(3)
    return T


def discretize(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order-hold discretization, exact for linear time-invariant (A, B).

    Uses expm([[A, B], [0, 0]] dt) = [[A_d, B_d], [0, I]].

    Args:
        A: Continuous state matrix, shape (n, n).
        B: Continuous input matrix, shape (n, m).
        dt: Sample time, s.

    Returns:
        A_d, shape (n, n), and B_d, shape (n, m).
    """
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"dt must be positive and finite, got {dt}")
    n, m = B.shape
    if A.shape != (n, n):
        raise ValueError(f"A must have shape ({n}, {n}), got {A.shape}")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
        raise ValueError("A and B must be finite")

    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    M_d = expm(M * dt)
    return M_d[:n, :n], M_d[:n, n:]
