"""K-MPC: deterministic Koopman MPC with mean wind, Eqs. (15)-(16).

Posed in physical units, solved in the model's coordinates:

    min  Σ_{k=1}^{np} e_kᵀ Q e_k + Σ_{k=0}^{np-1} ν_kᵀ R ν_k
    s.t. ψ_{k+1} = A ψ_k + B ν_k + D v̄_w,   ψ_0 = ψ((e_0 − μ) / σ)
         e_k = σ ⊙ (C ψ_k) + μ
         u_lb ≤ ν_k ≤ u_ub,   h_lb ≤ e_k ≤ h_ub

The model acts on the standardized error z = (e − μ)/σ (koopman.preprocessing),
so with S = diag(σ), e_kᵀ Q e_k = ‖C ψ_k − r‖²_{SQS} with r = −μ/σ: zero
physical error is the output reference r, not C ψ = 0. Inputs and wind are
physical in the model (B, D act on them directly), so R and the input bounds
need no conversion. Terminal weight P = Q, as in the paper. The output ν_0 is
added to the LQR command as the perturbation u_d.

Units and frames (datagen.collect.tracking_error, controllers.sdlqr_model):
    e_x = [p, v, roll/pitch/yaw] error, m, m/s, rad; p and v in world NED.
    u_d = [ω_x, ω_y, ω_z (body FRD, rad/s), c (mass-normalized thrust, m/s²)].
    v̄_w: mean wind, world NED, m/s.

The solve is treated as instantaneous: ν_0 is applied at the sample it was
computed for, with no delay compensation.
"""
from dataclasses import dataclass
import time

import numpy as np

from dynamics._validation import as_finite_vector

from .qp import condense, solve_qp


@dataclass(frozen=True)
class KMPCSettings:
    """Weights, horizon, bounds and mean wind, all in physical units.

    Attributes:
        Q: (n, n) symmetric PSD weight on e_k, k = 1..np (terminal included).
        R: (m, m) symmetric PD weight on u_d.
        horizon: n_p ≥ 1.
        u_lb, u_ub: (m,) bounds on the perturbation u_d (not on u_h + u_d);
            ±inf for unbounded channels.
        h_lb, h_ub: (n,) bounds on e_k, k = 1..np; ±inf for unbounded channels.
        wind_mean: (d,) v̄_w, world NED, m/s.
    """

    Q: np.ndarray
    R: np.ndarray
    horizon: int
    u_lb: np.ndarray
    u_ub: np.ndarray
    h_lb: np.ndarray
    h_ub: np.ndarray
    wind_mean: np.ndarray

    def __post_init__(self) -> None:
        for name in ("Q", "R", "wind_mean"):
            value = np.asarray(getattr(self, name), dtype=float)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains NaN or inf")
            object.__setattr__(self, name, value)
        for name in ("u_lb", "u_ub", "h_lb", "h_ub"):
            value = np.asarray(getattr(self, name), dtype=float)
            if np.any(np.isnan(value)):
                raise ValueError(f"{name} contains NaN")
            object.__setattr__(self, name, value)
        for lo, hi in (("u_lb", "u_ub"), ("h_lb", "h_ub")):
            lower, upper = getattr(self, lo), getattr(self, hi)
            if lower.ndim != 1 or lower.shape != upper.shape:
                raise ValueError(f"{lo} and {hi} must be equal 1-D shapes, got "
                                 f"{lower.shape} and {upper.shape}")
            if np.any(lower > upper):
                raise ValueError(f"{lo} must not exceed {hi}")


@dataclass(frozen=True)
class MPCStep:
    """Result of one K-MPC solve.

    Attributes:
        u_d: (m,) perturbation to apply; zeros on fallback.
        solved: False if the QP failed and u_d fell back to 0 (pure LQR).
        exitflag: DAQP status code (1 optimal, -1 infeasible).
        iterations: Active-set iterations.
        solve_time: Wall-clock time of the whole step (standardize, lift,
            QP data, QP solve), s; what counts against the dt_augment budget.
        planned_inputs: (np, m) ν_0..ν_{np-1} behind u_d; zeros on fallback.
        predicted_error: (np, n) e_1..e_np predicted for planned_inputs,
            physical units.
    """

    u_d: np.ndarray
    solved: bool
    exitflag: int
    iterations: int
    solve_time: float
    planned_inputs: np.ndarray
    predicted_error: np.ndarray


class KoopmanMPC:
    """K-MPC on a Koopman Model 1, Eq. (5); see the module docstring.

    The condensed QP is built once here; each solve only lifts the measured
    error and updates the linear cost term and the state-constraint bounds.
    The previous plan, shifted by one step, warm-starts the next solve.
    """

    def __init__(self, model, lifting, standardizer, settings: KMPCSettings):
        """
        Args:
            model: KoopmanModel (Model 1, use_residual=False) on z = (e − μ)/σ.
            lifting: ψ used to fit the model; state_dim n, lifted_dim N.
            standardizer: Standardizer the model was fitted with.
            settings: KMPCSettings in physical units.
        """
        if model.use_residual:
            raise ValueError("K-MPC supports Model 1 only; Model 2 needs u_res")
        N, m, d = model.A.shape[0], model.B.shape[1], model.D.shape[1]
        n = model.C.shape[0]
        if lifting.state_dim != n or lifting.lifted_dim != N:
            raise ValueError(f"lifting maps {lifting.state_dim} -> {lifting.lifted_dim}, "
                             f"model expects {n} -> {N}")
        if standardizer.mean.shape != (n,):
            raise ValueError(f"standardizer has {standardizer.mean.shape[0]} channels, "
                             f"model has {n}")
        for name, size in (("u_lb", m), ("h_lb", n), ("wind_mean", d)):
            shape = getattr(settings, name).shape
            if shape != (size,):
                raise ValueError(f"settings.{name} must have shape ({size},), got {shape}")

        self.lifting = lifting
        self.standardizer = standardizer
        self.settings = settings
        self._C = model.C
        sigma, mu = standardizer.std, standardizer.mean
        Q_z = sigma[:, None] * settings.Q * sigma[None, :]  # S Q S
        self.qp = condense(model.A, model.B, model.D, model.C, Q_z, settings.R,
                           settings.horizon)
        self._reference = -mu / sigma
        self._h_lb = (settings.h_lb - mu) / sigma  # ±inf stays ±inf, σ > 0
        self._h_ub = (settings.h_ub - mu) / sigma
        self._u_lb = np.tile(settings.u_lb, settings.horizon)
        self._u_ub = np.tile(settings.u_ub, settings.horizon)
        self._m = m
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        """Forget the warm start, e.g. between episodes."""
        self._previous = None

    def solve(self, e_x) -> MPCStep:
        """One K-MPC step from the measured tracking error e_x, shape (n,), physical units."""
        start = time.perf_counter()
        n = self._C.shape[0]
        e_x = as_finite_vector("e_x", e_x, n)
        psi0 = self.lifting(self.standardizer.transform(e_x)[None, :])[0]
        v_w = self.settings.wind_mean

        f = self.qp.linear_term(psi0, v_w, self._reference)
        g_lb, g_ub = self.qp.state_bounds(psi0, v_w, self._h_lb, self._h_ub)
        result = solve_qp(self.qp.H, f, lb=self._u_lb, ub=self._u_ub,
                          G=self.qp.G, g_lb=g_lb, g_ub=g_ub, x0=self._warm_start())

        V = result.x if result.solved else np.zeros(self.qp.num_vars)
        self._previous = V if result.solved else None
        predicted = self.standardizer.inverse(self.qp.predict(psi0, V, v_w) @ self._C.T)
        planned = V.reshape(self.settings.horizon, self._m)
        return MPCStep(u_d=planned[0].copy(), solved=result.solved,
                       exitflag=result.exitflag, iterations=result.iterations,
                       solve_time=time.perf_counter() - start, planned_inputs=planned,
                       predicted_error=predicted)

    def _warm_start(self) -> np.ndarray | None:
        """Previous plan shifted by one step, last input repeated."""
        if self._previous is None:
            return None
        return np.concatenate([self._previous[self._m:], self._previous[-self._m:]])
