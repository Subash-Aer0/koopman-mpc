"""Low-level body-rate controller: u_h = [ω_des, c] -> body thrust and torque.

The fast inner loop of [16], Sec. II-A: the LQR treats body rates as an input,
which assumes this loop is much faster than the LQR closed loop.

Feedback linearization of Eq. (4) of the rigid body:
    τ_t = J K_ω (ω_des − ω) + ω × (J ω)
cancels the gyroscopic term, so each axis obeys ω̇ = k_ω,i (ω_des,i − ω_i), a
first-order lag with time constant 1 / k_ω,i. Thrust is passed through:
    f_t = [0, 0, −m c]        (body FRD, thrust along −z_body)

Outputs (f_t, τ_t) go straight to dynamics.rk4_step; the rotor mixer and
rotor model (rotor.py) will later sit between this controller and the body.
"""
import numpy as np

from dynamics._validation import as_finite_vector
from dynamics.dynamics import RATE as X_RATE
from dynamics.dynamics import STATE_DIM
from dynamics.params import QuadParams

from .sdlqr_model import INPUT_DIM, RATE, THRUST


class BodyRateController:
    """Body-rate tracking with gyroscopic compensation."""

    def __init__(self, params: QuadParams, gains):
        """
        Args:
            params: Vehicle parameters; uses mass (kg) and inertia J (kg·m², body FRD).
            gains: K_ω diagonal, body FRD axes, 1/s, shape (3,), positive.
                1 / k_ω,i is the rate-loop time constant of axis i.
        """
        gains = as_finite_vector("gains", gains, 3)
        if np.any(gains <= 0):
            raise ValueError(f"gains must be positive, got {gains}")
        self.params = params
        self.gains = gains

    def __call__(self, x, u_h) -> tuple[np.ndarray, np.ndarray]:
        """Thrust and torque that track u_h.

        Args:
            x: Full rigid-body state, shape (13,); layout in dynamics.py.
                Only the body rates ω (rad/s, body FRD) are used.
            u_h: High-level command [ω_des (rad/s, body FRD), c (m/s²)], shape (4,).

        Returns:
            f_t: Total thrust, body FRD, N, shape (3,).
            τ_t: Total torque, body FRD, N·m, shape (3,).
        """
        x = as_finite_vector("x", x, STATE_DIM)
        u_h = as_finite_vector("u_h", u_h, INPUT_DIM)
        omega = x[X_RATE]
        J = self.params.inertia

        tau_t = J @ (self.gains * (u_h[RATE] - omega)) + np.cross(omega, J @ omega)
        f_t = np.array([0.0, 0.0, -self.params.mass * u_h[THRUST]])
        return f_t, tau_t
