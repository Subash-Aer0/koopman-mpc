"""High-level state-dependent LQR tracking controller (Foehn & Scaramuzza [16]).

Outputs u_h = [ω_x, ω_y, ω_z, c] for the low-level controller; designed
without wind knowledge. State, input, frames and units follow sdlqr_model.py:
x = [p, q, v] (NED world, quaternion body FRD -> world NED, scalar first),
u = [ω (body FRD, rad/s), c (mass-normalized thrust along −z_body, m/s²)].

The gain is re-solved at the most recent state (10 Hz in [16], Sec. III-B)
and applied at a faster rate (100 Hz in [16]) via u = u_ref + K (x − x_ref),
Eq. (15) of [16].

Deviation from [16]: the discrete Riccati equation is solved in the 9-dim
tangent space of the unit quaternion (sdlqr_model.tangent_basis) instead of by
dynamic programming on the 10-dim state. The norm direction of q is
uncontrollable with an eigenvalue on the unit circle, so the 10-dim DARE has
no stabilizing solution; the reduced one does and gives the same feedback on
every physically meaningful error.
"""
import numpy as np
from scipy.linalg import solve_discrete_are

from dynamics._validation import as_finite_vector

from .sdlqr_model import (
    INPUT_DIM,
    QUAT,
    RATE,
    STATE_DIM,
    THRUST,
    discretize,
    linearize,
    tangent_basis,
)


def _validate_weight(name: str, M, size: int, positive_definite: bool) -> np.ndarray:
    M = np.array(M, dtype=float)
    if M.shape != (size, size) or not np.all(np.isfinite(M)):
        raise ValueError(f"{name} must be a finite {size}x{size} matrix, got shape {M.shape}")
    if not np.allclose(M, M.T):
        raise ValueError(f"{name} must be symmetric")
    eig_min = np.linalg.eigvalsh(M).min()
    if eig_min < 0 or (positive_definite and eig_min == 0):
        kind = "positive definite" if positive_definite else "positive semi-definite"
        raise ValueError(f"{name} must be {kind}, smallest eigenvalue {eig_min}")
    return M


def saturate(u: np.ndarray, max_body_rate: float,
             thrust_limits: tuple[float, float]) -> np.ndarray:
    """Scale ω onto the ‖ω‖ <= max_body_rate ball (keeps the rotation axis) and clip c.

    Args:
        u: Command [ω (rad/s, body FRD), c (m/s²)], shape (4,).
        max_body_rate: Bound on ‖ω‖, rad/s.
        thrust_limits: (c_min, c_max), m/s².

    Returns:
        Saturated copy of u, shape (4,).
    """
    u = np.array(u, dtype=float)
    rate_norm = np.linalg.norm(u[RATE])
    if rate_norm > max_body_rate:
        u[RATE] *= max_body_rate / rate_norm
    u[THRUST] = np.clip(u[THRUST], *thrust_limits)
    return u


def lqr_gain(x, u, Q: np.ndarray, R: np.ndarray, dt: float) -> np.ndarray:
    """Discrete infinite-horizon LQR gain linearized at (x, u), Eqs. (14)-(15) of [16].

    Linearize, project onto the tangent basis T(q), discretize with ZOH, solve
    the DARE in 9 dims, and map back: K = K̃ Tᵀ. Inputs are not validated here;
    LQRTracker validates them once at its public boundary.

    Args:
        x: Linearization state [p, q, v], shape (10,).
        u: Linearization input [ω, c], shape (4,).
        Q: State cost, shape (10, 10), symmetric PSD.
        R: Input cost, shape (4, 4), symmetric PD.
        dt: Control sample time, s.

    Returns:
        K, shape (4, 10), for u = u_ref + K (x − x_ref). K q = 0 along the
        quaternion norm direction.

    Raises:
        numpy.linalg.LinAlgError: If the reduced DARE has no stabilizing solution.
    """
    A, B = linearize(x, u)
    T = tangent_basis(x[QUAT])
    A_d, B_d = discretize(T.T @ A @ T, T.T @ B, dt)
    Q_r = T.T @ Q @ T

    P = solve_discrete_are(A_d, B_d, Q_r, R)
    K_r = -np.linalg.solve(R + B_d.T @ P @ B_d, B_d.T @ P @ A_d)
    return K_r @ T.T


class LQRTracker:
    """State-dependent LQR that tracks a reference state with feedforward input.

    Usage per control step: call update_gain(x, u) at the gain rate, then
    __call__(x, x_ref, u_ref) at the control rate.
    """

    def __init__(self, Q, R, dt: float, max_body_rate: float,
                 thrust_limits: tuple[float, float]):
        """
        Args:
            Q: State cost, shape (10, 10), symmetric PSD, order [p, q, v].
            R: Input cost, shape (4, 4), symmetric PD, order [ω, c].
            dt: Control sample time used to discretize the model, s.
            max_body_rate: Bound on ‖ω‖, rad/s.
            thrust_limits: (c_min, c_max), mass-normalized thrust, m/s².
        """
        if not (np.isfinite(dt) and dt > 0):
            raise ValueError(f"dt must be positive and finite, got {dt}")
        if not (np.isfinite(max_body_rate) and max_body_rate > 0):
            raise ValueError(f"max_body_rate must be positive and finite, got {max_body_rate}")
        c_min, c_max = thrust_limits
        if not (np.isfinite(c_min) and np.isfinite(c_max) and 0 <= c_min < c_max):
            raise ValueError(f"thrust_limits must satisfy 0 <= c_min < c_max, got {thrust_limits}")

        self.Q = _validate_weight("Q", Q, STATE_DIM, positive_definite=False)
        self.R = _validate_weight("R", R, INPUT_DIM, positive_definite=True)
        self.dt = float(dt)
        self.max_body_rate = float(max_body_rate)
        self.thrust_limits = (float(c_min), float(c_max))
        self.K: np.ndarray | None = None

    def update_gain(self, x, u) -> np.ndarray:
        """Re-linearize at (x, u) and store the new gain K, shape (4, 10)."""
        x = as_finite_vector("x", x, STATE_DIM)
        u = as_finite_vector("u", u, INPUT_DIM)
        self.K = lqr_gain(x, u, self.Q, self.R, self.dt)
        return self.K

    def __call__(self, x, x_ref, u_ref) -> np.ndarray:
        """Saturated command u_h = sat(u_ref + K (x − x_ref)).

        Args:
            x: Current state [p, q, v], shape (10,).
            x_ref: Reference state, shape (10,). Its quaternion sign is flipped
                if needed so that q and q_ref describe the shortest rotation.
            u_ref: Feedforward input [ω, c], shape (4,).

        Returns:
            u_h = [ω (rad/s), c (m/s²)], shape (4,), with ‖ω‖ <= max_body_rate
            and c within thrust_limits.
        """
        if self.K is None:
            raise RuntimeError("update_gain must be called before the first command")
        x = as_finite_vector("x", x, STATE_DIM)
        x_ref = as_finite_vector("x_ref", x_ref, STATE_DIM).copy()
        u_ref = as_finite_vector("u_ref", u_ref, INPUT_DIM)

        # q and −q are the same rotation; without this the error can encode a 360° turn.
        if np.dot(x[QUAT], x_ref[QUAT]) < 0:
            x_ref[QUAT] = -x_ref[QUAT]

        u = u_ref + self.K @ (x - x_ref)
        return self.saturate(u)

    def saturate(self, u) -> np.ndarray:
        """Apply this tracker's input limits to any command u, shape (4,).

        Used on u_h + u_d so an augmenting term cannot exceed the limits.
        """
        u = as_finite_vector("u", u, INPUT_DIM)
        return saturate(u, self.max_body_rate, self.thrust_limits)
