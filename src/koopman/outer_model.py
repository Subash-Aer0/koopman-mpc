"""Koopman model of the open-loop outer-loop plant (command -> vehicle state), for full MPC.

Unlike koopman.models (the paper's LQR-in-the-loop error model), this model
maps the *total* 10 Hz command to the vehicle state, so an MPC can replace the
LQR. Coordinates are deviations from the hover trim in mean wind shared with
the physics baseline (controllers.hover_model):

    s = [p, v, η − η*, ω] / σ          (12 states; ω are measured body rates)
    z = ψ(s)                            (TrigMonomialLifting; no trig angles = monomials only)
    z⁺ = A z + B δu + D δv_w,   δu = u − u*,  δv_w = v_w − v̄_w
    δx (9 physical deviations) = C z,   C = [diag(σ_1..9)  0]

The state is scaled but not shifted (σ only), so z = 0 at the trim except for
the constant feature. Body rates are included so the model can learn the
rate-loop lag that the physics baseline ignores (ideal rate tracking); they
are not costed. Frames and units as hover_model: world NED, ZYX Euler, body
FRD rates (rad/s), c in m/s².
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from dynamics import dynamics as rigid_body
from dynamics._validation import as_finite_vector

from .trig_lifting import TrigMonomialLifting

STATE_NAMES = ("p_x", "p_y", "p_z", "v_x", "v_y", "v_z", "roll", "pitch", "yaw",
               "w_x", "w_y", "w_z")
PHYSICAL_DIM = 9
EULER = slice(6, 9)


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def deviation_state(x9, omega, x_trim, include_rates: bool) -> np.ndarray:
    """[p, v, η, (ω)] − trim, angles wrapped; x9 (T, 9), omega (T, 3) -> (T, 9 or 12)."""
    x9 = np.atleast_2d(np.asarray(x9, dtype=float))
    dx = x9 - np.asarray(x_trim, dtype=float)
    dx[:, EULER] = wrap_angle(dx[:, EULER])
    if include_rates:
        dx = np.hstack([dx, np.atleast_2d(np.asarray(omega, dtype=float))])
    return dx


@dataclass(frozen=True)
class OuterKoopmanModel:
    """Identified lifted model plus everything needed to lift a measurement."""

    A: np.ndarray
    B: np.ndarray
    D: np.ndarray
    scale: np.ndarray            # σ per state channel, physical units
    degree: int
    nonlinear_indices: tuple
    include_rates: bool
    x_trim: np.ndarray           # (9,)
    u_trim: np.ndarray           # (4,)
    wind_mean: np.ndarray        # (3,) world NED, m/s
    dt: float
    trig_indices: tuple = ()     # state indices of angles lifted as sin, cos − 1

    @property
    def state_dim(self) -> int:
        return 12 if self.include_rates else PHYSICAL_DIM

    @property
    def lifting(self) -> TrigMonomialLifting:
        trig = list(self.trig_indices)
        return TrigMonomialLifting(self.state_dim, self.degree, self.nonlinear_indices, trig,
                                   self.scale[trig] if trig else None)

    @property
    def C(self) -> np.ndarray:
        """(9, N): physical deviation [p, v, η] − trim from the lifted state."""
        C = np.zeros((PHYSICAL_DIM, self.A.shape[0]))
        C[:, :PHYSICAL_DIM] = np.diag(self.scale[:PHYSICAL_DIM])
        return C

    def lift_deviation(self, dx) -> np.ndarray:
        """Deviation states (T, n) -> lifted states (T, N)."""
        return self.lifting(np.atleast_2d(dx) / self.scale)

    def lift_measurement(self, x13) -> np.ndarray:
        """13-state rigid-body x -> z (N,), for the MPC."""
        from controllers.linear_mpc import outer_state  # local: avoid a package cycle
        x13 = as_finite_vector("x", x13, rigid_body.STATE_DIM)
        dx = deviation_state(outer_state(x13), x13[rigid_body.RATE], self.x_trim,
                             self.include_rates)
        return self.lift_deviation(dx)[0]

    def save(self, directory) -> None:
        directory = Path(directory)
        np.savez(directory / "outer_model.npz", A=self.A, B=self.B, D=self.D, scale=self.scale,
                 x_trim=self.x_trim, u_trim=self.u_trim, wind_mean=self.wind_mean)
        (directory / "outer_model.json").write_text(json.dumps({
            "degree": self.degree, "nonlinear_indices": list(self.nonlinear_indices),
            "include_rates": self.include_rates, "dt": self.dt,
            "trig_indices": list(self.trig_indices)}, indent=2))

    @classmethod
    def load(cls, directory) -> "OuterKoopmanModel":
        directory = Path(directory)
        spec = json.loads((directory / "outer_model.json").read_text())
        with np.load(directory / "outer_model.npz") as data:
            arrays = {key: data[key] for key in data.files}
        return cls(**arrays, degree=int(spec["degree"]),
                   nonlinear_indices=tuple(spec["nonlinear_indices"]),
                   include_rates=bool(spec["include_rates"]), dt=float(spec["dt"]),
                   trig_indices=tuple(spec.get("trig_indices", ())))   # older runs: none


def horizon_rmse(A, B, D, C, lift, episodes, horizon: int) -> np.ndarray:
    """RMSE of open-loop predictions per step ahead and output channel.

    From every start k of every episode, lift once and iterate
    z_{j+1} = A z_j + B δu_{k+j} + D δw_{k+j} without re-lifting (as the MPC
    does), and compare C z_j with the measured output.

    Args:
        A, B, D, C: Linear (lifted) model and output map.
        lift: Deviation states (T, n) -> lifted states (T, N).
        episodes: List of dicts with "dx" (T, n) model states, "y" (T, n_y)
            measured outputs, "du" (T, 4), "dw" (T, 3).
        horizon: H, steps ahead.

    Returns:
        (H, n_y) RMSE in the units of y; row j is j + 1 steps ahead.
    """
    sq, count = 0.0, 0
    for ep in episodes:
        T = len(ep["y"])
        if T <= horizon:
            continue
        starts = T - horizon
        z = lift(ep["dx"][:starts])
        errors = []
        for j in range(horizon):
            z = z @ A.T + ep["du"][j:starts + j] @ B.T + ep["dw"][j:starts + j] @ D.T
            errors.append(z @ C.T - ep["y"][j + 1:starts + j + 1])
        sq = sq + np.sum(np.stack(errors) ** 2, axis=1)
        count += starts
    if count == 0:
        raise ValueError(f"no episode is longer than horizon {horizon}")
    return np.sqrt(sq / count)
