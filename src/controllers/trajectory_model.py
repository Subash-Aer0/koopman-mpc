"""Linearization of the outer-loop model along a reference trajectory, for LTV MPC.

The hover model (hover_model.py) expands the 9-state model about one trim. Here
the expansion point moves with the reference: at every sample k the point
(x_ref,k, u_ref,k) is a trajectory of the *nonlinear* model, and

    e_{k+1} = A_k e_k + B_k Δ_k + D_k δv_w + r_k,    e = x − x_ref,k,  Δ = u − u_ref,k

with (A_k, B_k, D_k) the Jacobians of the one-sample map F (nonlinear model,
input and wind held for dt) at that point, and r_k the one-step residual of the
sampled reference (below).

Why the Jacobians of F and not expm of the continuous Jacobians at t_k (as
hover_model.discretize does): at a trim the state does not move during a
sample, so the two coincide. Along a trajectory the attitude moves within the
sample, and freezing the Jacobian at t_k is a first-order error in e (about
2–3 % of A − I and of B on the 5 s circle). Linearizing the map the MPC
predicts with ("discretize, then linearize") leaves only second-order error.

Same state, input, frames and units as hover_model.py:
    x = [p (m), v (m/s), η = (φ, θ, ψ) (rad)], world NED, ZYX Euler, body FRD
    u = [ω (body FRD, rad/s), c (mass-normalized thrust along −z_body, m/s²)]
    v_w: mean wind, world NED, m/s, held constant (assumption A4).
Yaw is held constant along the reference (assumption A3); "constant yaw" means
the ZYX Euler angle ψ, which is what the MPC error state measures.

Reference point (the "trim along the path"). Given (p, v, a, j)(t) and ψ:
    attitude and thrust: c b₃ = g e₃ − a − f_D(R(η), v − v̄_w) / m,   b₃ = R(η) e₃
        Drag depends on the attitude, so (φ, θ, c) is found by fixed-point
        iteration, as in hover_trim (which is the case v = a = 0).
    rates: differentiate g(φ, θ, c; v, a) := v̇_model − a = 0 along the path
        (implicit function theorem):
            M [φ̇, θ̇, ċ]ᵀ = j − A_vv a,     M = [∂v̇/∂φ  ∂v̇/∂θ  ∂v̇/∂c]
        then ω = W(φ, θ)⁻¹ [φ̇, θ̇, 0]ᵀ. This is the inversion of design report
        §5.2 with the Jacobians taken at the reference point, not at hover.

Residual. The reference input varies within a sample but the MPC holds it
(ZOH), so the sampled reference is not exactly a discrete trajectory:
    r_k = F(x_ref,k, u_ref,k) − x_ref,k+1,   F = one ZOH step of the nonlinear model.
Keeping r_k in the prediction makes it exact on the reference (e = 0, Δ = 0).
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from dynamics._validation import as_finite_vector
from dynamics.drag import drag_force
from dynamics.params import QuadParams

from .hover_model import (
    EULER, PITCH, ROLL, THRUST, VEL, WIND_DIM,
    euler_dynamics, euler_rate_matrix, linearize, rotation_zyx,
)

# p_ref(t) -> (p, v, a, j), world NED, m, m/s, m/s², m/s³ (as linear_mpc.PositionDerivatives).
PositionDerivatives = Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]

_MAX_ITERATIONS = 100
_ATTITUDE_TOL = 1e-13    # change in the thrust direction b₃ between iterations
_MIN_THRUST = 1e-6       # m/s²; below this the thrust direction is undefined (free fall)
# RK4 sub-steps per sample for the residual. At dt = 0.1 s a sub-step of 0.01 s
# gives an integration error far below the residual itself (~dt² u̇).
_RESIDUAL_SUBSTEPS = 10
# Central-difference step for the Jacobians of F; same reasoning as hover_model._FD_STEP
# (smooth, O(1) map: truncation ~h² against round-off ~eps/h).
_FD_STEP = 1e-6


def _attitude_and_thrust(v, a, yaw: float, wind_mean, params: QuadParams):
    """(φ, θ, c) with c R(φ, θ, ψ) e₃ = g e₃ − a − f_D/m, by fixed-point iteration."""
    R_yaw = rotation_zyx([0.0, 0.0, yaw])
    euler = np.array([0.0, 0.0, yaw])
    b3_previous = None
    for _ in range(_MAX_ITERATIONS):
        f_D = drag_force(rotation_zyx(euler), v - wind_mean, params.drag_coeffs)
        thrust_vector = params.gravity_vector - a - f_D / params.mass   # = c R e₃
        c = np.linalg.norm(thrust_vector)
        if c < _MIN_THRUST:
            raise ValueError("reference demands free fall; thrust direction is undefined")
        b3 = thrust_vector / c
        # Remove yaw, then read roll and pitch from R e₃ = [s_θ c_φ, −s_φ, c_θ c_φ].
        bx, by, bz = R_yaw.T @ b3
        euler = np.array([np.arctan2(-by, np.hypot(bx, bz)), np.arctan2(bx, bz), yaw])
        if b3_previous is not None and np.max(np.abs(b3 - b3_previous)) < _ATTITUDE_TOL:
            return euler, c
        b3_previous = b3
    raise ValueError("reference attitude did not converge")


def trajectory_point(p, v, a, j, yaw: float, wind_mean,
                     params: QuadParams) -> tuple[np.ndarray, np.ndarray]:
    """Nonlinear reference point (x_ref, u_ref) for one sample of a position trajectory.

    Args:
        p, v, a, j: Position (m), velocity (m/s), acceleration (m/s²), jerk
            (m/s³), world NED, shape (3,) each.
        yaw: Constant ZYX heading ψ, rad.
        wind_mean: v̄_w, world NED, m/s, shape (3,).
        params: Vehicle parameters.

    Returns:
        x_ref = [p, v, φ, θ, ψ] (9,) and u_ref = [ω (rad/s), c (m/s²)] (4,), such
        that euler_dynamics(x_ref, u_ref, v̄_w) = [v, a, η̇_ref].

    Raises:
        ValueError: On non-finite inputs, free fall, gimbal lock, or if the
            attitude cannot be solved for.
    """
    p, v, a, j = (as_finite_vector(name, value, 3)
                  for name, value in (("p", p), ("v", v), ("a", a), ("j", j)))
    wind_mean = as_finite_vector("wind_mean", wind_mean, WIND_DIM)
    if not np.isfinite(yaw):
        raise ValueError(f"yaw must be finite, got {yaw}")
    euler, c = _attitude_and_thrust(v, a, yaw, wind_mean, params)
    x_ref = np.concatenate([p, v, euler])

    # The velocity rows of the Jacobians do not depend on ω, so ω = 0 is exact here.
    A_c, B_c, _ = linearize(x_ref, np.array([0.0, 0.0, 0.0, c]), wind_mean, params)
    M = np.column_stack([A_c[VEL, ROLL], A_c[VEL, PITCH], B_c[VEL, THRUST]])
    roll_dot, pitch_dot, _ = np.linalg.solve(M, j - A_c[VEL, VEL] @ a)
    W = euler_rate_matrix(euler[0], euler[1])
    omega = np.linalg.solve(W, np.array([roll_dot, pitch_dot, 0.0]))
    return x_ref, np.concatenate([omega, [c]])


def _zoh_step(x, u, wind_mean, params: QuadParams, dt: float) -> np.ndarray:
    """One sample of the 9-state nonlinear model with u held, by RK4 sub-steps."""
    h = dt / _RESIDUAL_SUBSTEPS
    for _ in range(_RESIDUAL_SUBSTEPS):
        k1 = euler_dynamics(x, u, wind_mean, params)
        k2 = euler_dynamics(x + h / 2 * k1, u, wind_mean, params)
        k3 = euler_dynamics(x + h / 2 * k2, u, wind_mean, params)
        k4 = euler_dynamics(x + h * k3, u, wind_mean, params)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def _step_jacobians(x, u, wind_mean, params: QuadParams, dt: float):
    """(A, B, D) = ∂F/∂(x, u, v_w) of the one-sample map at (x, u, v̄_w), central differences."""
    def jacobian(func, point):
        columns = []
        for i in range(len(point)):
            step = np.zeros(len(point))
            step[i] = _FD_STEP
            columns.append((func(point + step) - func(point - step)) / (2 * _FD_STEP))
        return np.column_stack(columns)

    A = jacobian(lambda z: _zoh_step(z, u, wind_mean, params, dt), x)
    B = jacobian(lambda w: _zoh_step(x, w, wind_mean, params, dt), u)
    D = jacobian(lambda w: _zoh_step(x, u, w, params, dt), wind_mean)
    return A, B, D


@dataclass(frozen=True)
class LTVModel:
    """Discrete model along a reference, sampled at t_k = k dt, k = 0..K−1.

    Attributes:
        t (K,): sample times, s.
        x_ref (K, 9), u_ref (K, 4): nonlinear reference points.
        A (K, 9, 9), B (K, 9, 4), D (K, 9, 3): Jacobians of the one-sample map at each point.
        r (K−1, 9): one-step residuals r_k = F(x_ref,k, u_ref,k) − x_ref,k+1
            (angles wrapped).
        wind_mean (3,): v̄_w, world NED, m/s.
        yaw: constant reference heading ψ, rad.
        dt: sample time, s.
    """

    t: np.ndarray
    x_ref: np.ndarray
    u_ref: np.ndarray
    A: np.ndarray
    B: np.ndarray
    D: np.ndarray
    r: np.ndarray
    wind_mean: np.ndarray
    yaw: float
    dt: float

    @classmethod
    def build(cls, derivatives: PositionDerivatives, duration: float, params: QuadParams,
              wind_mean, yaw: float, dt: float) -> "LTVModel":
        """Sample the reference over [0, duration] and linearize the one-sample map at each sample.

        Args:
            derivatives: (p, v, a, j)(t) of the position reference.
            duration: Time span to cover, s; the last sample is at ≥ duration.
            params: Vehicle parameters.
            wind_mean: v̄_w, world NED, m/s, shape (3,).
            yaw: Constant ZYX heading ψ, rad.
            dt: Sample time, s (the MPC period).
        """
        if not (np.isfinite(dt) and dt > 0):
            raise ValueError(f"dt must be positive and finite, got {dt}")
        if not (np.isfinite(duration) and duration > 0):
            raise ValueError(f"duration must be positive and finite, got {duration}")
        wind_mean = as_finite_vector("wind_mean", wind_mean, WIND_DIM)
        count = int(np.ceil(duration / dt - 1e-9)) + 1   # tolerance: duration = k dt exactly
        t = dt * np.arange(count)
        points = [trajectory_point(*derivatives(tk), yaw, wind_mean, params) for tk in t]
        x_ref = np.array([x for x, _ in points])
        u_ref = np.array([u for _, u in points])
        A, B, D = (np.array(m) for m in zip(*(
            _step_jacobians(x, u, wind_mean, params, dt) for x, u in points)))
        r = np.array([_zoh_step(x_ref[k], u_ref[k], wind_mean, params, dt) - x_ref[k + 1]
                      for k in range(count - 1)])
        r[:, EULER] = (r[:, EULER] + np.pi) % (2 * np.pi) - np.pi
        return cls(t=t, x_ref=x_ref, u_ref=u_ref, A=A, B=B, D=D, r=r,
                   wind_mean=wind_mean, yaw=float(yaw), dt=float(dt))

