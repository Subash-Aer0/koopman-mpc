"""Actuator between the rate controller and the rigid body: commanded wrench -> applied wrench.

    (f_t, τ_t) commanded ─► mixer (quadratic model) ─► Ω_cmd ─► motor ─► Ω ─► rotor model in local airflow ─► (f_t, τ_t) applied

The rate controller computes the thrust and torque it wants assuming the
nominal quadratic rotor model; the mixer turns that into rotor speeds Ω. The
rotor model then decides what the rotors really produce:
    "ideal"  the commanded wrench exactly (no rotor model; the behaviour before
             rotors existed in the simulator)
    "bemt"   BEMT thrust and torque per rotor in its own airflow
             v_i = Rᵀ(v − v_w) + ω × r_i (dynamics/rotor.py)
Motor (optional, "bemt" only, dynamics/motor.py): without one, Ω = Ω_cmd (ideal
motor). With one, Ω has its own dynamics and the aerodynamic torque Q_BEMT slows
it, so the rotor speeds a real vehicle would measure, Ω_R = Ω, differ from the
command; the CCTA paper's residual thrust f_res is computed from them.
This is truth (plant) only: no controller sees it, so a mismatch with the
nominal model appears only in airflow.

State: with a motor the actuator holds Ω. Call reset(dt_sim) before every
simulation (the simulators do); each call then advances Ω by one dt_sim.

The wrench is evaluated once per simulation step and held over it, like the
rate controller's output.

Frames: body FRD, thrust along −z_body; v, v_w world NED, m/s; ω body FRD, rad/s.
"""
from pathlib import Path

import numpy as np

from ._validation import as_finite_vector
from .dynamics import QUAT, RATE, STATE_DIM, VEL
from .quaternion import Quaternion
from .motor import MotorParams, load_motor_params, motor_step
from .rotor import (N_ROTORS, RotorParams, bemt_thrust, body_wrench, load_rotor_params, mixer,
                    rotor_airspeeds)

MODELS = ("ideal", "bemt")


class RotorActuator:
    """Callable (x, f_t, τ_t, v_w) -> applied (f_t, τ_t)."""

    def __init__(self, model: str, rotor: RotorParams | None = None,
                 config_paths: tuple[Path, ...] = (), motor: MotorParams | None = None):
        """
        Args:
            model: One of MODELS.
            rotor: Rotor parameters; required unless model is "ideal".
            config_paths: Files the rotor parameters came from, for run records.
            motor: Motor parameters; None for an ideal motor (Ω = Ω_cmd). Needs "bemt".
        """
        if model not in MODELS:
            raise ValueError(f"actuator model must be one of {MODELS}, got {model!r}")
        if model != "ideal" and rotor is None:
            raise ValueError(f"actuator model {model!r} needs rotor parameters")
        if motor is not None and model != "bemt":
            raise ValueError("a motor model needs the 'bemt' rotor model (it is loaded by Q_BEMT)")
        self.model = model
        self.rotor = rotor
        self.motor = motor
        self.config_paths = tuple(Path(p) for p in config_paths)
        self._dt: float | None = None
        self._omega: np.ndarray | None = None          # motor state Ω, rad/s
        self._speeds = np.full(N_ROTORS, np.nan)      # Ω behind the latest wrench

    def reset(self, dt: float) -> None:
        """Start a new simulation with step dt (s): forget the motor state."""
        if not (np.isfinite(dt) and dt > 0):
            raise ValueError(f"dt must be positive and finite, got {dt}")
        self._dt = float(dt)
        self._omega = None
        self._speeds = np.full(N_ROTORS, np.nan)

    @property
    def rotor_speeds(self) -> np.ndarray:
        """Ω (4,) rad/s at which the latest wrench was evaluated: what a vehicle measures.

        Equals the mixer command without a motor; NaN for "ideal" (no rotor model)
        and before the first call.
        """
        return self._speeds.copy()

    def __call__(self, x, f_t, tau_t, v_w) -> tuple[np.ndarray, np.ndarray]:
        """Applied thrust (3,) N and torque (3,) N·m, body FRD, for the 13-state x."""
        f_t = as_finite_vector("f_t", f_t, 3)
        tau_t = as_finite_vector("tau_t", tau_t, 3)
        if self.model == "ideal":
            return f_t, tau_t
        x = as_finite_vector("x", x, STATE_DIM)
        v_w = as_finite_vector("v_w", v_w, 3)
        if not (f_t[0] == 0.0 and f_t[1] == 0.0):
            raise ValueError(f"rotors only thrust along −z_body, got f_t = {f_t}")
        R = Quaternion(x[QUAT]).normalize().to_rotation_matrix()
        omega_cmd = mixer(-f_t[2], tau_t, self.rotor)
        airspeeds = rotor_airspeeds(R.T @ (x[VEL] - v_w), x[RATE], self.rotor)
        if self.motor is None:
            omega = omega_cmd
        else:
            if self._dt is None:
                raise RuntimeError("call reset(dt_sim) before simulating with a motor model")
            if self._omega is None:
                self._omega = omega_cmd.copy()       # start at the first command
            omega = self._omega
        T, Q, _ = bemt_thrust(omega, airspeeds, self.rotor)
        self._speeds = omega.copy()
        if self.motor is not None:
            self._omega = motor_step(omega, omega_cmd, Q, self.rotor.torque_coefficient,
                                     self.motor, self._dt)
        return body_wrench(T, Q, self.rotor)


def load_actuator(vehicle_cfg: dict, vehicle_path: str | Path) -> RotorActuator:
    """Actuator from a vehicle config's [rotor] (and optional [motor]) tables.

    [rotor] model = "ideal" | "bemt", rotor_config = path relative to the vehicle file;
    "ideal" when the table is absent. [motor] model = "ideal" | "first_order" with
    time_constant (s) and rotor_inertia (kg·m²); ideal motor when the table is absent.
    """
    section = vehicle_cfg.get("rotor")
    if section is None:
        return RotorActuator("ideal")
    model = str(section.get("model", ""))
    motor = load_motor_params(vehicle_cfg.get("motor"))
    if model == "ideal":
        if motor is not None:
            raise ValueError("[motor] needs [rotor] model = 'bemt'")
        return RotorActuator("ideal")
    if "rotor_config" not in section:
        raise ValueError(f"[rotor] model = {model!r} needs rotor_config")
    rotor_path = (Path(vehicle_path).resolve().parent / section["rotor_config"]).resolve()
    return RotorActuator(model, load_rotor_params(rotor_path), (rotor_path,), motor=motor)
