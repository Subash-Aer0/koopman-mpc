"""Error-state LTV MPC: the linear MPC re-linearized along the reference.

Same problem as linear_mpc.LinearMPC, with the hover model replaced by the
model along the reference (trajectory_model.LTVModel). At the sample k = t/dt:

    min  Σ_{i=1}^{N-1} e_iᵀ Q e_i + e_Nᵀ P_{k+N} e_N + Σ_{i=0}^{N-1} Δ_iᵀ R Δ_i
    s.t. e_{i+1} = A_{k+i} e_i + B_{k+i} Δ_i + r_{k+i},   e_0 = x − x_ref,k measured
         u_lb ≤ u_ref,k+i + Δ_i ≤ u_ub                       (total command)
         |φ|, |θ| ≤ max_tilt at x_ref,k+i + e_i,  i = 1..N   (total attitude)
         terminal:  "cost_only"  P_{k+N} only
                    "equality"   e_N = 0
P_k comes from the backward Riccati recursion over the whole grid, started
from the DARE of the last sample. The recursion forgets its end condition
(riccati.py), so the last `riccati_lead` seconds of the grid only serve to make
P_{k+N} converge and are never used as a horizon. A "polytope" terminal set
would need invariant sets along the trajectory and is not implemented.

The residual r enters the condensed QP as a disturbance D_i = r_{k+i} of
value 1 (qp.condense_ltv). If a solve is infeasible, the step uses the
input-bound-only problem and is logged, as in LinearMPC.

Units, frames and the 13-state input are those of linear_mpc.py.
"""
import numpy as np
from scipy.linalg import solve_discrete_are

from .hover_model import EULER, INPUT_DIM, PITCH, ROLL, STATE_DIM
from .linear_mpc import LinearMPCSettings, MPCLog, outer_state, wrap_angle
from .qp import condense_ltv, solve_qp
from .riccati import time_varying_riccati
from .trajectory_model import LTVModel

TERMINAL_MODES = ("cost_only", "equality")
_AFFINE = np.ones(1)          # the "disturbance" multiplying the residual columns
_TIME_TOL = 1e-6              # s; a call must fall on the model's sample grid


class LTVMPC:
    """Error-state LTV MPC; call with (t, 13-state x) every `period` seconds."""

    def __init__(self, model: LTVModel, settings: LinearMPCSettings, riccati_lead: float):
        """
        Args:
            model: Model along the reference; its dt is the MPC period. Its
                grid must cover the run, one horizon, and the lead.
            settings: Weights, horizon, terminal mode, limits (as LinearMPC,
                so both baselines are configured identically).
            riccati_lead: Length, s, of the grid's tail used only to converge
                the Riccati recursion.

        Raises:
            ValueError: For terminal="polytope", a negative lead, a grid too
                short for one horizon after the lead, or a reference that uses
                more input or tilt than the limits allow.
        """
        if settings.terminal not in TERMINAL_MODES:
            raise ValueError(f"LTVMPC supports terminal modes {TERMINAL_MODES}, "
                             f"got {settings.terminal!r}")
        if not (np.isfinite(riccati_lead) and riccati_lead >= 0):
            raise ValueError(f"riccati_lead must be non-negative, got {riccati_lead}")
        self.model = model
        self.settings = settings
        self.period = model.dt
        N, K = settings.horizon, len(model.t)
        lead_steps = int(np.ceil(riccati_lead / model.dt - 1e-9))
        self.last_index = K - 1 - lead_steps - N   # the last sample a solve may start at
        if self.last_index < 0:
            raise ValueError(f"model grid of {K} samples is too short for horizon {N} "
                             f"and a {riccati_lead} s Riccati lead")

        reachable = slice(0, self.last_index + N + 1)
        if np.any(model.u_ref[reachable] < settings.u_lb) \
                or np.any(model.u_ref[reachable] > settings.u_ub) \
                or np.any(np.abs(model.x_ref[reachable][:, [ROLL, PITCH]]) > settings.max_tilt):
            raise ValueError("the reference uses more input or tilt than the limits allow")

        P_end = solve_discrete_are(model.A[-1], model.B[-1], settings.Q, settings.R)
        self.P, self.K = time_varying_riccati(model.A[:-1], model.B[:-1],
                                              settings.Q, settings.R, P_end)

        self._tilt_rows = np.zeros((2, STATE_DIM))
        self._tilt_rows[0, ROLL] = 1.0
        self._tilt_rows[1, PITCH] = 1.0
        self._cond = [self._condense(k) for k in range(self.last_index + 1)]
        self.log = MPCLog()
        self._previous = None

    def _condense(self, k: int):
        """Condensed QP for a horizon starting at sample k."""
        N, model = self.settings.horizon, self.model
        window = slice(k, k + N)
        return condense_ltv(model.A[window], model.B[window], model.r[window][:, :, None],
                            np.eye(STATE_DIM), self.settings.Q, self.settings.R,
                            P=self.P[k + N], E=self._tilt_rows)

    @property
    def fallback_count(self) -> int:
        return int(np.sum(self.log.fallback))

    def index(self, t: float) -> int:
        """Grid sample k for time t, s; raises if t is off the grid or past the last solve."""
        k = int(round(t / self.model.dt))
        if abs(t - k * self.model.dt) > _TIME_TOL:
            raise ValueError(f"t = {t} s is not on the {self.model.dt} s sample grid")
        if not 0 <= k <= self.last_index:
            raise ValueError(f"t = {t} s is outside the model's usable range "
                             f"[0, {self.last_index * self.model.dt}] s")
        return k

    def error_state(self, t: float, x) -> np.ndarray:
        """e = x₉ − x_ref,k, angles wrapped; x is the 13-state."""
        e = outer_state(x) - self.model.x_ref[self.index(t)]
        e[EULER] = wrap_angle(e[EULER])
        return e

    def __call__(self, t: float, x) -> np.ndarray:
        """Total command u = [ω, c] for the 13-state x at time t."""
        s, N = self.settings, self.settings.horizon
        k = self.index(t)
        cond = self._cond[k]
        e0 = self.error_state(t, x)
        u_ref = self.model.u_ref[k:k + N]
        tilt_ref = self.model.x_ref[k + 1:k + N + 1][:, [ROLL, PITCH]]

        f = cond.linear_term(e0, _AFFINE)
        lb = (s.u_lb - u_ref).ravel()
        ub = (s.u_ub - u_ref).ravel()
        g_lb, g_ub = cond.state_bounds(e0, _AFFINE, -s.max_tilt - tilt_ref,
                                       s.max_tilt - tilt_ref)
        G, rows_lb, rows_ub = [cond.G], [g_lb], [g_ub]
        if s.terminal == "equality":
            # e_N = Φ_N e_0 + Γ_N V + Λ_N = 0
            free_N = (cond.Phi @ e0 + cond.Lambda @ _AFFINE)[-STATE_DIM:]
            G.append(cond.Gamma[-STATE_DIM:])
            rows_lb.append(-free_N)
            rows_ub.append(-free_N)

        warm = None
        if self._previous is not None:  # shifted previous plan, last move repeated
            warm = np.concatenate([self._previous[INPUT_DIM:], self._previous[-INPUT_DIM:]])
        result = solve_qp(cond.H, f, lb, ub, np.vstack(G), np.concatenate(rows_lb),
                          np.concatenate(rows_ub), x0=warm)
        fallback = not result.solved
        if fallback:
            result = solve_qp(cond.H, f, lb, ub)
            if not result.solved:
                raise RuntimeError(f"input-bound-only MPC failed at t = {t} s "
                                   f"(exitflag {result.exitflag})")
        self._previous = result.x
        u = u_ref[0] + result.x[:INPUT_DIM]
        # The QP enforces the bounds up to solver tolerance; clip only that round-off.
        u = np.clip(u, s.u_lb, s.u_ub)

        self.log.t.append(t)
        self.log.fallback.append(fallback)
        self.log.exitflag.append(result.exitflag)
        self.log.solve_time.append(result.solve_time)
        self.log.error.append(e0)
        self.log.command.append(u)
        return u
