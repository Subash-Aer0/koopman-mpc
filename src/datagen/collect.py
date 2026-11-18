"""Closed-loop data collection with the LQR in the loop (Fig. 1, Sec. IV-A).

At 10 Hz, record: tracking error e_x = x_c − x_r (position, velocity, rpy),
random perturbations u_d added to u_h, and wind v_w. These are the signals of
Model 1, Eq. (5). The residual thrust f_res of Model 2 needs measured rotor
speeds Ω_R and is not collected until dynamics/rotor.py exists.

Perturbations are drawn up front (pre-generated, Sec. IV-B) so Model 2 can
later reuse exactly the same u_d sequence.
"""
from dataclasses import dataclass, fields

import numpy as np

from controllers import sdlqr_model as lqr_model
from dynamics import dynamics as rigid_body
from dynamics.quaternion import Quaternion
from simulation.closed_loop import SimulationLog

ERROR_DIM = 9
# Slices of the tracking-error vector e_x = [p (m), v (m/s), rpy (rad)].
ERR_POS = slice(0, 3)
ERR_VEL = slice(3, 6)
ERR_RPY = slice(6, 9)

# Tolerance on t / period being an integer, as in simulation.closed_loop.
_ALIGN_TOL = 1e-9


def random_perturbations(n_steps: int, bounds, rng: np.random.Generator) -> np.ndarray:
    """Uniform u_d,k ~ U(−a, a) per channel, i.i.d. over k.

    Args:
        n_steps: Number of augmentation periods.
        bounds: a = [ω_x, ω_y, ω_z (rad/s), c (m/s²)], shape (4,), each ≥ 0.
        rng: Source of the draws.

    Returns:
        u_d, shape (n_steps, 4), input order of controllers/sdlqr_model.py.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (lqr_model.INPUT_DIM,) or not np.all(np.isfinite(bounds)):
        raise ValueError(f"bounds must be {lqr_model.INPUT_DIM} finite numbers, got {bounds}")
    if np.any(bounds < 0):
        raise ValueError(f"bounds must be non-negative, got {bounds}")
    if not (isinstance(n_steps, (int, np.integer)) and n_steps > 0):
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    return rng.uniform(-bounds, bounds, size=(n_steps, lqr_model.INPUT_DIM))


class HeldPerturbations:
    """Augmentation hook replaying a pre-generated u_d sequence, value k on [k·T, (k+1)·T)."""

    def __init__(self, values, period: float):
        """
        Args:
            values: u_d, shape (n, 4).
            period: T, s; must equal the simulator's dt_augment.
        """
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != lqr_model.INPUT_DIM:
            raise ValueError(f"values must have shape (n, {lqr_model.INPUT_DIM}), "
                             f"got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("values contain NaN or inf")
        if not (np.isfinite(period) and period > 0):
            raise ValueError(f"period must be positive and finite, got {period}")
        self.values = values
        self.period = float(period)

    def __call__(self, t: float, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        ratio = t / self.period
        k = round(ratio)
        if abs(ratio - k) > _ALIGN_TOL * max(1.0, ratio):
            raise ValueError(f"t = {t} s is not a multiple of the period {self.period} s; "
                             "period must equal the simulator's dt_augment")
        if not 0 <= k < len(self.values):
            raise ValueError(f"t = {t} s is past the {len(self.values)} pre-generated "
                             "perturbations")
        return self.values[k].copy()


def tracking_error(x, x_ref) -> np.ndarray:
    """e_x = x_c − x_r as [position, velocity, roll/pitch/yaw] (Sec. III-A).

    The attitude error is the difference of ZYX Euler angles, wrapped to
    [−π, π); it is ill-defined near pitch ±π/2 (see Quaternion.to_euler).

    Args:
        x: Vehicle states, shape (N, 13), layout in dynamics.py.
        x_ref: Reference states, shape (N, 10), layout in sdlqr_model.py.

    Returns:
        e_x, shape (N, 9): m, m/s, rad; NED world frame for p and v.
    """
    x = np.asarray(x, dtype=float)
    x_ref = np.asarray(x_ref, dtype=float)
    if x.ndim != 2 or x.shape[1] != rigid_body.STATE_DIM:
        raise ValueError(f"x must have shape (N, {rigid_body.STATE_DIM}), got {x.shape}")
    if x_ref.shape != (len(x), lqr_model.STATE_DIM):
        raise ValueError(f"x_ref must have shape ({len(x)}, {lqr_model.STATE_DIM}), "
                         f"got {x_ref.shape}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(x_ref))):
        raise ValueError("x or x_ref contains NaN or inf")

    def euler(quaternions: np.ndarray) -> np.ndarray:
        return np.array([Quaternion(q).normalize().to_euler() for q in quaternions])

    e_x = np.empty((len(x), ERROR_DIM))
    e_x[:, ERR_POS] = x[:, rigid_body.POS] - x_ref[:, lqr_model.POS]
    e_x[:, ERR_VEL] = x[:, rigid_body.VEL] - x_ref[:, lqr_model.VEL]
    d_rpy = euler(x[:, rigid_body.QUAT]) - euler(x_ref[:, lqr_model.QUAT])
    e_x[:, ERR_RPY] = (d_rpy + np.pi) % (2 * np.pi) - np.pi
    return e_x


@dataclass(frozen=True)
class Split:
    """E episodes of T samples at dt_log, stacked along a leading episode axis.

    Keeping episodes separate (not concatenated) is deliberate: snapshot
    pairs (e_x[i, k], e_x[i, k+1]) must never span two episodes.

    Attributes:
        t: Sample times, s, shape (T,), shared by all episodes.
        reference: Reference name per episode, shape (E,).
        repeat: Repeat index per episode within its reference, shape (E,).
        e_x: Tracking error, shape (E, T, 9), see tracking_error.
        u_d: Perturbation held over [t_k, t_k+1), shape (E, T, 4).
        v_w: Wind at t_k, NED world, m/s, shape (E, T, 3).
        x: Vehicle state, shape (E, T, 13); raw, so stage 2 can redefine e_x.
        x_ref: Reference state, shape (E, T, 10).
        u_h: LQR command (saturated), shape (E, T, 4).
        u_cmd: sat(u_h + u_d) sent to the low-level loop, shape (E, T, 4).
    """

    t: np.ndarray
    reference: np.ndarray
    repeat: np.ndarray
    e_x: np.ndarray
    u_d: np.ndarray
    v_w: np.ndarray
    x: np.ndarray
    x_ref: np.ndarray
    u_h: np.ndarray
    u_cmd: np.ndarray

    def __post_init__(self) -> None:
        n_episodes, n_samples = len(self.reference), len(self.t)
        if self.repeat.shape != (n_episodes,):
            raise ValueError(f"repeat must have shape ({n_episodes},), got {self.repeat.shape}")
        widths = {"e_x": ERROR_DIM, "u_d": lqr_model.INPUT_DIM, "v_w": 3,
                  "x": rigid_body.STATE_DIM, "x_ref": lqr_model.STATE_DIM,
                  "u_h": lqr_model.INPUT_DIM, "u_cmd": lqr_model.INPUT_DIM}
        for name, width in widths.items():
            value = getattr(self, name)
            if value.shape != (n_episodes, n_samples, width):
                raise ValueError(f"{name} must have shape ({n_episodes}, {n_samples}, {width}), "
                                 f"got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains NaN or inf")

    @property
    def n_pairs(self) -> int:
        """Snapshot pairs (k, k+1) available inside episodes."""
        return len(self.reference) * (len(self.t) - 1)


ARRAY_FIELDS = tuple(f.name for f in fields(Split))


def split_from_logs(logs: list[SimulationLog], references: list[str],
                    repeats: list[int]) -> Split:
    """Stack equal-length simulation logs into a Split, computing e_x per episode."""
    if not logs:
        raise ValueError("need at least one log")
    if not len(logs) == len(references) == len(repeats):
        raise ValueError("logs, references and repeats must have equal lengths")
    t = logs[0].t
    for log in logs[1:]:
        if not np.array_equal(log.t, t):
            raise ValueError("all logs must share the same sample times")
    return Split(
        t=t.copy(),
        reference=np.array(references, dtype=str),
        repeat=np.array(repeats, dtype=int),
        e_x=np.stack([tracking_error(log.x, log.x_ref) for log in logs]),
        u_d=np.stack([log.u_d for log in logs]),
        v_w=np.stack([log.v_w for log in logs]),
        x=np.stack([log.x for log in logs]),
        x_ref=np.stack([log.x_ref for log in logs]),
        u_h=np.stack([log.u_h for log in logs]),
        u_cmd=np.stack([log.u_cmd for log in logs]),
    )
