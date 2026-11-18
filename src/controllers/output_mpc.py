"""Output-tracking linear MPC in the form of [LN-DDC] Eq. (17), for any lifted linear model.

    min  Σ_{i=0}^{N-1} (‖C z_i − r_i‖²_Q + ‖δu_i‖²_R) + ‖C z_N − r_N‖²_Q
    s.t. z_{i+1} = A z_i + B δu_i,   z_0 = ψ(x),   u_lb ≤ u* + δu_i ≤ u_ub

The cost acts on the physical output C z (not on the lifted state); the
terminal weight is Q (no terminal set, no Riccati terminal cost) and only the
input is constrained, exactly as in the course notes. The z_0 term is constant
and dropped. Deviations δu = u − u* are from the hover trim in mean wind, which
both models share, so the mean wind needs no separate term (δv_w = 0 is
predicted; turbulence is not).

The same class runs the physics baseline (z = x − x*, C = I) and the Koopman
model (z = ψ(scaled x − x*)), so the two differ only in (A, B, C, ψ).

Units: C z and r in the physical deviation coordinates of hover_model
([p (m), v (m/s), η (rad)] − trim); u = [ω (rad/s), c (m/s²)].
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from dynamics._validation import as_finite_vector

from .hover_model import INPUT_DIM
from .linear_mpc import MPCLog
from .qp import condense, solve_qp


@dataclass(frozen=True)
class OutputMPCSettings:
    """Q (n_y, n_y) PSD on C z − r, R (4, 4) PD on δu, horizon N ≥ 1, total-command bounds."""

    Q: np.ndarray
    R: np.ndarray
    horizon: int
    u_lb: np.ndarray
    u_ub: np.ndarray

    def __post_init__(self) -> None:
        Q = np.asarray(self.Q, dtype=float)
        R = np.asarray(self.R, dtype=float)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or not np.allclose(Q, Q.T):
            raise ValueError("Q must be a symmetric square matrix")
        if np.min(np.linalg.eigvalsh(Q)) < -1e-12:
            raise ValueError("Q must be positive semidefinite")
        if R.shape != (INPUT_DIM, INPUT_DIM) or not np.allclose(R, R.T) \
                or np.min(np.linalg.eigvalsh(R)) <= 0:
            raise ValueError(f"R must be symmetric positive definite {INPUT_DIM}x{INPUT_DIM}")
        if not (isinstance(self.horizon, (int, np.integer)) and self.horizon >= 1):
            raise ValueError(f"horizon must be a positive integer, got {self.horizon!r}")
        u_lb = as_finite_vector("u_lb", self.u_lb, INPUT_DIM)
        u_ub = as_finite_vector("u_ub", self.u_ub, INPUT_DIM)
        if np.any(u_lb >= u_ub):
            raise ValueError("u_lb must be below u_ub in every channel")
        for name, value in (("Q", Q), ("R", R), ("u_lb", u_lb), ("u_ub", u_ub)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "horizon", int(self.horizon))


class OutputMPC:
    """Eq. (17) MPC; call with (t, 13-state x) every `period` seconds."""

    def __init__(self, A, B, C, lift: Callable[[np.ndarray], np.ndarray], u_trim,
                 settings: OutputMPCSettings, reference: Callable[[float], np.ndarray],
                 period: float):
        """
        Args:
            A (N, N), B (N, 4): discrete model z⁺ = A z + B δu at sample time `period`.
            C (n_y, N): physical output y = C z, deviation from trim.
            lift: 13-state x -> z (N,).
            u_trim: (4,) u*, the command the deviations are taken from.
            settings: Weights, horizon, total-command bounds.
            reference: t -> r (n_y,), the output target in the same coordinates as C z.
            period: Sample time, s.
        """
        A = np.asarray(A, dtype=float)
        n = A.shape[0]
        C = np.asarray(C, dtype=float)
        if C.shape != (settings.Q.shape[0], n):
            raise ValueError(f"C must have shape ({settings.Q.shape[0]}, {n}), got {C.shape}")
        self.settings = settings
        self.C = C
        self.lift = lift
        self.reference = reference
        self.period = float(period)
        self.u_trim = as_finite_vector("u_trim", u_trim, INPUT_DIM)
        self._cond = condense(A, B, np.zeros((n, 1)), C, settings.Q, settings.R,
                              settings.horizon, P=settings.Q)
        N = settings.horizon
        self._lb = np.tile(settings.u_lb - self.u_trim, N)
        self._ub = np.tile(settings.u_ub - self.u_trim, N)
        if np.any(self._lb >= 0) or np.any(self._ub <= 0):
            raise ValueError("the trim command is outside the input bounds")
        self.log = MPCLog()
        self._previous = None

    def __call__(self, t: float, x) -> np.ndarray:
        s, N = self.settings, self.settings.horizon
        z0 = self.lift(x)
        r = np.array([self.reference(t + i * self.period) for i in range(1, N + 1)])
        f = self._cond.linear_term(z0, np.zeros(1), r)
        warm = None
        if self._previous is not None:
            warm = np.concatenate([self._previous[INPUT_DIM:], self._previous[-INPUT_DIM:]])
        result = solve_qp(self._cond.H, f, self._lb, self._ub, x0=warm)
        if not result.solved:  # box constraints with H ≻ 0 are always feasible
            raise RuntimeError(f"input-bounded MPC failed at t = {t} s (exitflag {result.exitflag})")
        self._previous = result.x
        # The QP enforces the bounds up to solver tolerance; clip only that round-off.
        u = np.clip(self.u_trim + result.x[:INPUT_DIM], s.u_lb, s.u_ub)
        self.log.t.append(t)
        self.log.fallback.append(False)
        self.log.exitflag.append(result.exitflag)
        self.log.solve_time.append(result.solve_time)
        self.log.error.append(self.C @ z0 - self.reference(t))
        self.log.command.append(u)
        return u


def output_reference(derivatives, x_trim) -> Callable[[float], np.ndarray]:
    """r(t) = [p_ref − p*, v_ref, 0, 0, 0]: position and velocity tracked, attitude at trim.

    Args:
        derivatives: t -> (p, v, a, j) of the reference, world NED (m, m/s, ...).
        x_trim: Hover trim x* (9,) the output deviations are taken from.

    Returns:
        t -> r(t), shape (9,), in the output coordinates of this module.
    """
    x_trim = as_finite_vector("x_trim", x_trim, 9)

    def reference(t: float) -> np.ndarray:
        p, v, _, _ = derivatives(t)
        return np.concatenate([np.asarray(p) - x_trim[:3], v, np.zeros(3)])
    return reference
