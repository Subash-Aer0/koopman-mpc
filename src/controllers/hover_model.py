"""Hover linearization of the outer-loop quadrotor model, for linear MPC.

The MPC treats body rates as an input tracked by the low-level loop (ideal rate
tracking, the same time-scale assumption as the LQR, [16] Sec. II-A), so the
model it predicts with has 9 states. Derivation: docs/koopman_mpc_study_guide.md
§10 and the hover-trim derivation it summarises.

State x = [p (3), v (3), η (3)]:
    p  position, world NED, m
    v  inertial velocity, world NED, m/s
    η  ZYX Euler angles [roll φ, pitch θ, yaw ψ], body FRD -> world NED, rad
       (the same convention as Quaternion.to_euler)
Input u = [ω (3), c]:
    ω  commanded body rates, body FRD, rad/s
    c  mass-normalized collective thrust along −z_body, m/s²
Disturbance v_w: wind velocity, world NED, m/s.

Model:
    ṗ = v
    v̇ = g e₃ − c R(η) e₃ − f_D(R, v − v_w) / m       (f_D from dynamics.drag)
    η̇ = W(φ, θ) ω

Linearized about a hover trim (x*, u*) in mean wind v̄_w:
    δẋ = A_c δx + B_c δu + D_c δv_w,   δx = x − x*, δu = u − u*, δv_w = v_w − v̄_w
and discretized with a zero-order hold on u and v_w.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from dynamics._validation import as_finite_vector
from dynamics.drag import drag_force
from dynamics.params import QuadParams

STATE_DIM = 9
INPUT_DIM = 4
WIND_DIM = 3
POS = slice(0, 3)
VEL = slice(3, 6)
EULER = slice(6, 9)
RATE = slice(0, 3)
THRUST = 3
ROLL, PITCH, YAW = 6, 7, 8

# Central-difference step for the Jacobians. The model is smooth and O(1)
# around hover, so 1e-6 balances truncation (~h²) against round-off (~eps/h).
_FD_STEP = 1e-6
_TRIM_MAX_ITERATIONS = 100
_TRIM_TOL = 1e-13         # change in the thrust direction between iterations
_MAX_TRIM_TILT = 0.5      # rad; a hover trim tilted more than this is outside this model's use


def rotation_zyx(euler) -> np.ndarray:
    """R = R_z(ψ) R_y(θ) R_x(φ), body FRD -> world NED, for η = [φ, θ, ψ] in rad."""
    phi, theta, psi = as_finite_vector("euler", euler, 3)
    c_f, s_f = np.cos(phi), np.sin(phi)
    c_t, s_t = np.cos(theta), np.sin(theta)
    c_p, s_p = np.cos(psi), np.sin(psi)
    return np.array([
        [c_p * c_t, c_p * s_t * s_f - s_p * c_f, c_p * s_t * c_f + s_p * s_f],
        [s_p * c_t, s_p * s_t * s_f + c_p * c_f, s_p * s_t * c_f - c_p * s_f],
        [-s_t, c_t * s_f, c_t * c_f],
    ])


def euler_rate_matrix(roll: float, pitch: float) -> np.ndarray:
    """W(φ, θ) with η̇ = W ω for ZYX Euler angles; singular at θ = ±π/2."""
    c_f, s_f = np.cos(roll), np.sin(roll)
    c_t, t_t = np.cos(pitch), np.tan(pitch)
    if abs(c_t) < 1e-6:
        raise ValueError(f"pitch {pitch} rad is at the ZYX gimbal lock")
    return np.array([
        [1.0, s_f * t_t, c_f * t_t],
        [0.0, c_f, -s_f],
        [0.0, s_f / c_t, c_f / c_t],
    ])


def euler_dynamics(x, u, v_w, params: QuadParams) -> np.ndarray:
    """ẋ of the 9-state model above; x (9,), u (4,), v_w (3,) -> (9,)."""
    x = as_finite_vector("x", x, STATE_DIM)
    u = as_finite_vector("u", u, INPUT_DIM)
    v_w = as_finite_vector("v_w", v_w, WIND_DIM)
    R = rotation_zyx(x[EULER])
    f_D = drag_force(R, x[VEL] - v_w, params.drag_coeffs)
    x_dot = np.empty(STATE_DIM)
    x_dot[POS] = x[VEL]
    x_dot[VEL] = params.gravity_vector - u[THRUST] * R[:, 2] - f_D / params.mass
    x_dot[EULER] = euler_rate_matrix(x[ROLL], x[PITCH]) @ u[RATE]
    return x_dot


def hover_trim(wind_mean, yaw: float, params: QuadParams) -> tuple[np.ndarray, np.ndarray]:
    """Hover equilibrium (x*, u*) at p = 0 and heading `yaw` in constant wind.

    At rest (v = 0, ω = 0) the thrust must cancel gravity and drag:
        c* R(η*) e₃ = g e₃ − f_D(R(η*), −v̄_w) / m.
    Drag is computed in body axes, so it depends on η*; the thrust direction is
    found by fixed-point iteration (converges in a few steps for small tilt).

    Args:
        wind_mean: v̄_w, world NED, m/s, shape (3,).
        yaw: ψ*, rad.
        params: Vehicle parameters.

    Returns:
        x* (9,) with p = 0, v = 0, η* = [φ*, θ*, yaw]; u* = [0, 0, 0, c*].

    Raises:
        ValueError: If the trim tilt exceeds _MAX_TRIM_TILT or iteration fails.
    """
    wind_mean = as_finite_vector("wind_mean", wind_mean, WIND_DIM)
    if not np.isfinite(yaw):
        raise ValueError(f"yaw must be finite, got {yaw}")
    R_yaw = rotation_zyx([0.0, 0.0, yaw])
    euler = np.array([0.0, 0.0, yaw])
    b3_previous = None
    for _ in range(_TRIM_MAX_ITERATIONS):
        f_D = drag_force(rotation_zyx(euler), -wind_mean, params.drag_coeffs)
        thrust_vector = params.gravity_vector - f_D / params.mass   # = c* R e₃
        c = np.linalg.norm(thrust_vector)
        b3 = thrust_vector / c
        # Remove yaw, then read roll and pitch from R e₃ = [s_θ c_φ, −s_φ, c_θ c_φ].
        bx, by, bz = R_yaw.T @ b3
        euler = np.array([np.arctan2(-by, np.hypot(bx, bz)), np.arctan2(bx, bz), yaw])
        if b3_previous is not None and np.max(np.abs(b3 - b3_previous)) < _TRIM_TOL:
            break
        b3_previous = b3
    else:
        raise ValueError("hover trim did not converge")
    if np.max(np.abs(euler[:2])) > _MAX_TRIM_TILT:
        raise ValueError(f"hover trim tilt {euler[:2]} rad exceeds {_MAX_TRIM_TILT} rad")
    x_star = np.concatenate([np.zeros(6), euler])
    u_star = np.array([0.0, 0.0, 0.0, c])
    return x_star, u_star


def linearize(x_star, u_star, wind_mean, params: QuadParams):
    """Continuous Jacobians (A_c, B_c, D_c) of euler_dynamics at (x*, u*, v̄_w).

    Central finite differences; tests compare them with the hand derivation.
    Returns A_c (9, 9) in 1/s and m/s²/rad etc., B_c (9, 4), D_c (9, 3).
    """
    x_star = as_finite_vector("x_star", x_star, STATE_DIM)
    u_star = as_finite_vector("u_star", u_star, INPUT_DIM)
    wind_mean = as_finite_vector("wind_mean", wind_mean, WIND_DIM)

    def jacobian(func, point):
        columns = []
        for i in range(len(point)):
            step = np.zeros(len(point))
            step[i] = _FD_STEP
            columns.append((func(point + step) - func(point - step)) / (2 * _FD_STEP))
        return np.column_stack(columns)

    A_c = jacobian(lambda x: euler_dynamics(x, u_star, wind_mean, params), x_star)
    B_c = jacobian(lambda u: euler_dynamics(x_star, u, wind_mean, params), u_star)
    D_c = jacobian(lambda w: euler_dynamics(x_star, u_star, w, params), wind_mean)
    return A_c, B_c, D_c


def discretize(A_c, B_c, D_c, dt: float):
    """Exact zero-order-hold discretization of (A_c, [B_c D_c]) at sample time dt, s.

    exp([[A_c, B_c, D_c], [0, 0, 0]] dt) = [[A, B, D], [0, I, 0], [0, 0, I]].
    """
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"dt must be positive and finite, got {dt}")
    n, m, d = A_c.shape[0], B_c.shape[1], D_c.shape[1]
    block = np.zeros((n + m + d, n + m + d))
    block[:n, :n] = A_c
    block[:n, n:n + m] = B_c
    block[:n, n + m:] = D_c
    E = expm(block * dt)
    return E[:n, :n], E[:n, n:n + m], E[:n, n + m:]


@dataclass(frozen=True)
class HoverModel:
    """Discrete hover linearization δx⁺ = A δx + B δu + D δv_w at sample time dt.

    Attributes:
        A (9, 9), B (9, 4), D (9, 3): discrete matrices.
        A_c, B_c, D_c: continuous Jacobians (kept for reference inversion).
        x_trim (9,), u_trim (4,): the trim the deviations are taken from.
        wind_mean (3,): v̄_w the trim was computed for, world NED, m/s.
        dt: Sample time, s.
    """

    A: np.ndarray
    B: np.ndarray
    D: np.ndarray
    A_c: np.ndarray
    B_c: np.ndarray
    D_c: np.ndarray
    x_trim: np.ndarray
    u_trim: np.ndarray
    wind_mean: np.ndarray
    dt: float

    @classmethod
    def build(cls, params: QuadParams, wind_mean, yaw: float, dt: float) -> "HoverModel":
        """Trim, linearize and discretize in one call."""
        x_star, u_star = hover_trim(wind_mean, yaw, params)
        A_c, B_c, D_c = linearize(x_star, u_star, wind_mean, params)
        A, B, D = discretize(A_c, B_c, D_c, dt)
        return cls(A=A, B=B, D=D, A_c=A_c, B_c=B_c, D_c=D_c, x_trim=x_star, u_trim=u_star,
                   wind_mean=np.array(wind_mean, dtype=float), dt=float(dt))
