"""13-state rigid-body quadcopter model in wind, after Eqs. (1)-(4).

State x = [p (3), q (4), v (3), ω (3)]:
    p  position of the centre of mass, NED world frame, m
    q  unit quaternion, body FRD -> world NED, scalar first (see quaternion.py)
    v  inertial (ground) velocity, NED world frame, m/s
    ω  angular rate, body FRD frame, rad/s

Inputs:
    f_t  total rotor thrust, body FRD frame, N. For a quadrotor f_t = [0, 0, −T].
    τ_t  total rotor torque about the centre of mass, body FRD frame, N·m.
    v_w  wind velocity, NED world frame, m/s.

Equations:
    ṗ = v
    q̇ = ½ q ⊗ [0, ω]                                (2)
    v̇ = g + (1/m) q ⊙ f_t − (1/m) f_D(v − v_w)       (3)
    ω̇ = J⁻¹ (τ_t − ω × (J ω))                       (4)
with g = [0, 0, g] (NED, z down).

Deviation from the paper: Eqs. (1) and (3) integrate the air-relative velocity
v_r and set ṗ = v_r + v_w, which drops the −v̇_w term and is exact only for
constant wind. Here the state is the inertial velocity v and drag acts on
v_r = v − v_w, which is physically consistent under turbulent wind and
identical to the paper when v_w is constant.

The rotor model (rotor.py) maps rotor speeds to (f_t, τ_t), so the rigid body
does not depend on which rotor model is used.
"""
import numpy as np

from ._validation import as_finite_vector
from .drag import drag_force
from .params import QuadParams
from .quaternion import Quaternion

STATE_DIM = 13
POS = slice(0, 3)
QUAT = slice(3, 7)
VEL = slice(7, 10)
RATE = slice(10, 13)

# Looser than quaternion.py's tolerance: RK4 stages evaluate the derivative at
# intermediate, slightly non-unit quaternions before the final renormalization.
_QUAT_NORM_TOL = 1e-3


def _validate_state(x) -> np.ndarray:
    x = as_finite_vector("x", x, STATE_DIM)
    norm = np.linalg.norm(x[QUAT])
    if abs(norm - 1.0) > _QUAT_NORM_TOL:
        raise ValueError(f"quaternion in x must be unit norm, got {norm}")
    return x


def _derivative(x: np.ndarray, f_t: np.ndarray, tau_t: np.ndarray,
                v_w: np.ndarray, params: QuadParams) -> np.ndarray:
    """Unchecked ẋ; callers validate inputs once at the public boundary."""
    q = Quaternion(x[QUAT]).normalize()
    v = x[VEL]
    omega = x[RATE]
    R = q.to_rotation_matrix()
    J = params.inertia

    f_D = drag_force(R, v - v_w, params.drag_coeffs)

    x_dot = np.empty(STATE_DIM)
    x_dot[POS] = v
    x_dot[QUAT] = 0.5 * Quaternion.multiply(q, Quaternion([0.0, *omega])).to_numpy()
    x_dot[VEL] = params.gravity_vector + (R @ f_t - f_D) / params.mass
    x_dot[RATE] = np.linalg.solve(J, tau_t - np.cross(omega, J @ omega))
    return x_dot


def state_derivative(x, f_t, tau_t, v_w, params: QuadParams) -> np.ndarray:
    """Continuous-time dynamics ẋ = f(x, f_t, τ_t, v_w).

    Args:
        x: State, shape (13,); layout and frames in the module docstring.
        f_t: Total thrust, body FRD, N, shape (3,).
        tau_t: Total torque, body FRD, N·m, shape (3,).
        v_w: Wind velocity, NED world, m/s, shape (3,).
        params: Vehicle parameters.

    Returns:
        ẋ, shape (13,), in the same units per second.
    """
    return _derivative(
        _validate_state(x),
        as_finite_vector("f_t", f_t, 3),
        as_finite_vector("tau_t", tau_t, 3),
        as_finite_vector("v_w", v_w, 3),
        params,
    )


def rk4_step(x, f_t, tau_t, v_w, params: QuadParams, dt: float) -> np.ndarray:
    """Advance the state by dt seconds with classical RK4.

    Inputs and wind are held constant over the step (zero-order hold). The
    quaternion is renormalized after the step to remove integration drift.
    """
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"dt must be positive and finite, got {dt}")
    x = _validate_state(x)
    f_t = as_finite_vector("f_t", f_t, 3)
    tau_t = as_finite_vector("tau_t", tau_t, 3)
    v_w = as_finite_vector("v_w", v_w, 3)

    k1 = _derivative(x, f_t, tau_t, v_w, params)
    k2 = _derivative(x + 0.5 * dt * k1, f_t, tau_t, v_w, params)
    k3 = _derivative(x + 0.5 * dt * k2, f_t, tau_t, v_w, params)
    k4 = _derivative(x + dt * k3, f_t, tau_t, v_w, params)
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    x_next[QUAT] = Quaternion(x_next[QUAT]).normalize().to_numpy()
    return x_next
