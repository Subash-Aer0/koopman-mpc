"""Multi-rate closed-loop simulation: LQR (+ optional augmentation) on the full model in wind.

Loop structure (Fig. 1 of the Koopman paper, rates from [16] Sec. III-B):

    reference ──x_ref,u_ref──► LQRTracker ──u_h──►(+)──sat──u_cmd──► BodyRateController ──f_t,τ_t──► rk4_step
                                   ▲                ▲                       ▲                           ▲
                          gain at dt_gain     u_d at dt_augment         every dt_sim               v_w = wind(t, p)

Commands are held between updates (zero-order hold). Every period is an integer
multiple of dt_sim, and the gain and augmentation periods are integer multiples
of dt_control, so all scheduling is done on integer step counters.

The augmentation hook is where data collection adds random perturbations and
where K-MPC / K-SMPC add their correction; the simulator does not know which.

Frames and units follow dynamics.py (13-state x) and controllers/sdlqr_model.py
(10-state x_ref, 4-dim inputs).
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from controllers import sdlqr_model as lqr_model
from controllers.low_level import BodyRateController
from controllers.lqr import LQRTracker
from dynamics import dynamics as rigid_body
from dynamics._validation import as_finite_vector
from dynamics.params import QuadParams

_STEP_RATIO_TOL = 1e-9


class Reference(Protocol):
    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return x_ref, shape (10,), and feedforward u_ref, shape (4,), at time t (s)."""


class Augmentation(Protocol):
    def __call__(self, t: float, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """Return u_d, shape (4,), added to u_h; x is the 13-state, x_ref the 10-state."""


class Wind(Protocol):
    def sample(self, t: float, position: np.ndarray) -> np.ndarray:
        """Return v_w, NED world frame, m/s, shape (3,)."""


def _steps(name: str, period: float, base: float) -> int:
    """period / base as a positive integer, or raise ValueError."""
    if not (np.isfinite(period) and period > 0):
        raise ValueError(f"{name} must be positive and finite, got {period}")
    ratio = period / base
    n = round(ratio)
    if n < 1 or abs(ratio - n) > _STEP_RATIO_TOL * max(1.0, ratio):
        raise ValueError(f"{name} = {period} s must be an integer multiple of {base} s")
    return n


@dataclass(frozen=True)
class Timing:
    """Loop periods, s. Defaults: 1 kHz sim, 100 Hz LQR, 10 Hz gain/augment/log.

    The 10 Hz augmentation and log rates match the Koopman sample time
    (config.DT = 0.1 s); the LQR rates follow [16], Sec. III-B.
    """

    dt_sim: float = 1e-3
    dt_control: float = 1e-2
    dt_gain: float = 0.1
    dt_augment: float = 0.1
    dt_log: float = 0.1

    def __post_init__(self) -> None:
        self.step_counts()

    def step_counts(self) -> dict[str, int]:
        """Each period in integration steps; raises if any ratio is not an integer."""
        _steps("dt_sim", self.dt_sim, self.dt_sim)
        n_control = _steps("dt_control", self.dt_control, self.dt_sim)
        return {
            "control": n_control,
            "gain": n_control * _steps("dt_gain", self.dt_gain, self.dt_control),
            "augment": n_control * _steps("dt_augment", self.dt_augment, self.dt_control),
            "log": _steps("dt_log", self.dt_log, self.dt_sim),
        }


@dataclass(frozen=True)
class SimulationLog:
    """Signals sampled every dt_log, all held values at that instant.

    Attributes:
        t: Time, s, shape (N,).
        x: Full state, shape (N, 13), layout in dynamics.py.
        x_ref: Reference state, shape (N, 10), layout in sdlqr_model.py.
        u_ref: Feedforward input, shape (N, 4).
        u_h: LQR command (saturated), shape (N, 4).
        u_d: Augmentation, shape (N, 4); zeros without augmentation.
        u_cmd: Command sent to the low-level loop, sat(u_h + u_d), shape (N, 4).
        v_w: Wind velocity, NED world, m/s, shape (N, 3).
        timing: Loop periods used.
    """

    t: np.ndarray
    x: np.ndarray
    x_ref: np.ndarray
    u_ref: np.ndarray
    u_h: np.ndarray
    u_d: np.ndarray
    u_cmd: np.ndarray
    v_w: np.ndarray
    timing: Timing


def lqr_state(x) -> np.ndarray:
    """Map the 13-state rigid-body x to the 10-state LQR x = [p, q, v] (drops ω)."""
    x = as_finite_vector("x", x, rigid_body.STATE_DIM)
    x_lqr = np.empty(lqr_model.STATE_DIM)
    x_lqr[lqr_model.POS] = x[rigid_body.POS]
    x_lqr[lqr_model.QUAT] = x[rigid_body.QUAT]
    x_lqr[lqr_model.VEL] = x[rigid_body.VEL]
    return x_lqr


def _reference(reference: Reference, t: float) -> tuple[np.ndarray, np.ndarray]:
    x_ref, u_ref = reference(t)
    return (as_finite_vector("x_ref", x_ref, lqr_model.STATE_DIM),
            as_finite_vector("u_ref", u_ref, lqr_model.INPUT_DIM))


def simulate(
    x0,
    duration: float,
    reference: Reference,
    wind: Wind,
    params: QuadParams,
    lqr: LQRTracker,
    rate_controller: BodyRateController,
    timing: Timing = Timing(),
    augmentation: Augmentation | None = None,
) -> SimulationLog:
    """Simulate the closed loop from x0 for `duration` seconds.

    Deterministic given its inputs; randomness lives in the wind and
    augmentation objects, which should be seeded by the caller.

    Args:
        x0: Initial 13-state, layout in dynamics.py.
        duration: Simulated time, s; an integer multiple of timing.dt_sim.
        reference: x_ref(t), u_ref(t).
        wind: v_w(t, p).
        params: Vehicle parameters of the simulated (true) plant.
        lqr: High-level controller; its dt must equal timing.dt_control.
        rate_controller: Low-level body-rate controller.
        timing: Loop periods.
        augmentation: Optional u_d(t, x, x_ref), held over dt_augment.

    Returns:
        SimulationLog sampled every timing.dt_log, starting at t = 0.
    """
    x = as_finite_vector("x0", x0, rigid_body.STATE_DIM)
    n_steps = _steps("duration", duration, timing.dt_sim)
    if not np.isclose(lqr.dt, timing.dt_control):
        raise ValueError(
            f"lqr.dt = {lqr.dt} s must equal timing.dt_control = {timing.dt_control} s; "
            "the gain is discretized with lqr.dt"
        )
    every = timing.step_counts()

    n_log = (n_steps - 1) // every["log"] + 1
    log = {
        "t": np.empty(n_log),
        "x": np.empty((n_log, rigid_body.STATE_DIM)),
        "x_ref": np.empty((n_log, lqr_model.STATE_DIM)),
        "u_ref": np.empty((n_log, lqr_model.INPUT_DIM)),
        "u_h": np.empty((n_log, lqr_model.INPUT_DIM)),
        "u_d": np.empty((n_log, lqr_model.INPUT_DIM)),
        "u_cmd": np.empty((n_log, lqr_model.INPUT_DIM)),
        "v_w": np.empty((n_log, 3)),
    }

    _, u_cmd = _reference(reference, 0.0)
    u_d = np.zeros(lqr_model.INPUT_DIM)
    for k in range(n_steps):
        t = k * timing.dt_sim
        is_control = k % every["control"] == 0
        is_log = k % every["log"] == 0

        if is_control or is_log:
            x_ref, u_ref = _reference(reference, t)
        if k % every["gain"] == 0:
            lqr.update_gain(lqr_state(x), u_cmd)
        if augmentation is not None and k % every["augment"] == 0:
            u_d = as_finite_vector("u_d", augmentation(t, x.copy(), x_ref.copy()),
                                   lqr_model.INPUT_DIM)
        if is_control:
            u_h = lqr(lqr_state(x), x_ref, u_ref)
            u_cmd = lqr.saturate(u_h + u_d)

        v_w = as_finite_vector("v_w", wind.sample(t, x[rigid_body.POS].copy()), 3)

        if is_log:
            i = k // every["log"]
            log["t"][i] = t
            log["x"][i] = x
            log["x_ref"][i] = x_ref
            log["u_ref"][i] = u_ref
            log["u_h"][i] = u_h
            log["u_d"][i] = u_d
            log["u_cmd"][i] = u_cmd
            log["v_w"][i] = v_w

        f_t, tau_t = rate_controller(x, u_cmd)
        x = rigid_body.rk4_step(x, f_t, tau_t, v_w, params, timing.dt_sim)

    return SimulationLog(**log, timing=timing)
