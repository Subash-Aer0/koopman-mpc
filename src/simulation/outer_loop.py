"""Closed loop with a full outer-loop controller in place of the LQR.

Same plant, rate loop, wind sampling and log format as closed_loop.simulate,
but the high-level command comes from any controller with the OuterLoopController
interface (e.g. controllers.linear_mpc.LinearMPC):

    controller(t, x) ──u_cmd (held over controller.period)──► BodyRateController ──► actuator ──► rk4_step
                                                                  ▲ every dt_sim         ▲ v_w = wind(t, p)

closed_loop.simulate is left unchanged, so the LQR experiments are unaffected.
The reference is only logged (for plotting and metrics); the controller owns
its own reference.

Frames and units: dynamics.py (13-state x), controllers/sdlqr_model.py (logged
x_ref, u_ref), u_cmd = [ω (body FRD, rad/s), c (m/s²)].
"""
from typing import Protocol

import numpy as np

from controllers import sdlqr_model as lqr_model
from controllers.low_level import BodyRateController
from dynamics import dynamics as rigid_body
from dynamics._validation import as_finite_vector
from dynamics.actuator import RotorActuator
from dynamics.params import QuadParams

from .closed_loop import Reference, SimulationLog, Timing, Wind


class OuterLoopController(Protocol):
    period: float  # s, command update period; an integer multiple of dt_sim

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        """Return u_cmd, shape (4,), for the 13-state x at time t (s)."""


def simulate_outer_loop(x0, duration: float, reference: Reference, wind: Wind,
                        params: QuadParams, controller: OuterLoopController,
                        rate_controller: BodyRateController, dt_sim: float = 1e-3,
                        dt_log: float = 0.1,
                        actuator: RotorActuator | None = None) -> SimulationLog:
    """Simulate from x0 for `duration` s with `controller` as the outer loop.

    actuator: the vehicle's rotors, commanded -> applied wrench (truth); None
    applies the command exactly (ideal rotors).

    Returns a SimulationLog sampled every dt_log with u_h = u_cmd (the
    controller's command) and u_d = 0, so all closed_loop plots and metrics apply.
    The log's Timing records the controller period as dt_control.
    """
    x = as_finite_vector("x0", x0, rigid_body.STATE_DIM)
    timing = Timing(dt_sim=dt_sim, dt_control=controller.period, dt_gain=controller.period,
                    dt_augment=controller.period, dt_log=dt_log)
    every = timing.step_counts()
    n_steps = round(duration / dt_sim)
    if n_steps < 1 or not np.isclose(n_steps * dt_sim, duration):
        raise ValueError(f"duration = {duration} s must be an integer multiple of {dt_sim} s")

    n_log = (n_steps - 1) // every["log"] + 1
    log = {
        "t": np.empty(n_log),
        "x": np.empty((n_log, rigid_body.STATE_DIM)),
        "x_ref": np.empty((n_log, lqr_model.STATE_DIM)),
        "u_ref": np.empty((n_log, lqr_model.INPUT_DIM)),
        "u_h": np.empty((n_log, lqr_model.INPUT_DIM)),
        "u_d": np.zeros((n_log, lqr_model.INPUT_DIM)),
        "u_cmd": np.empty((n_log, lqr_model.INPUT_DIM)),
        "v_w": np.empty((n_log, 3)),
    }
    u_cmd = None
    for k in range(n_steps):
        t = k * dt_sim
        if k % every["control"] == 0:
            u_cmd = as_finite_vector("u_cmd", controller(t, x.copy()), lqr_model.INPUT_DIM)
        v_w = as_finite_vector("v_w", wind.sample(t, x[rigid_body.POS].copy()), 3)
        if k % every["log"] == 0:
            i = k // every["log"]
            x_ref, u_ref = reference(t)
            log["t"][i] = t
            log["x"][i] = x
            log["x_ref"][i] = x_ref
            log["u_ref"][i] = u_ref
            log["u_h"][i] = u_cmd
            log["u_cmd"][i] = u_cmd
            log["v_w"][i] = v_w
        f_t, tau_t = rate_controller(x, u_cmd)
        if actuator is not None:
            f_t, tau_t = actuator(x, f_t, tau_t, v_w)
        x = rigid_body.rk4_step(x, f_t, tau_t, v_w, params, dt_sim)
    return SimulationLog(**log, timing=timing)
