"""Actuator between the rate controller and the rigid body: commanded wrench -> applied wrench.

    (f_t, τ_t) commanded ─► mixer (quadratic model) ─► Ω ─► rotor model in local airflow ─► (f_t, τ_t) applied

The rate controller computes the thrust and torque it wants assuming the
nominal quadratic rotor model; the mixer turns that into rotor speeds Ω. The
rotor model then decides what the rotors really produce:
    "ideal"  the commanded wrench exactly (no rotor model; the behaviour before
             rotors existed in the simulator)
    "bemt"   BEMT thrust and torque per rotor in its own airflow
             v_i = Rᵀ(v − v_w) + ω × r_i (dynamics/rotor.py)
This is truth (plant) only: no controller sees it, so a mismatch with the
nominal model appears only in airflow.

The wrench is evaluated once per simulation step and held over it, like the
rate controller's output.

Frames: body FRD, thrust along −z_body; v, v_w world NED, m/s; ω body FRD, rad/s.
"""
from pathlib import Path

import numpy as np

from ._validation import as_finite_vector
from .dynamics import QUAT, RATE, STATE_DIM, VEL
from .quaternion import Quaternion
from .rotor import RotorParams, bemt_thrust, body_wrench, load_rotor_params, mixer, rotor_airspeeds

MODELS = ("ideal", "bemt")


class RotorActuator:
    """Callable (x, f_t, τ_t, v_w) -> applied (f_t, τ_t)."""

    def __init__(self, model: str, rotor: RotorParams | None = None,
                 config_paths: tuple[Path, ...] = ()):
        """
        Args:
            model: One of MODELS.
            rotor: Rotor parameters; required unless model is "ideal".
            config_paths: Files the rotor parameters came from, for run records.
        """
        if model not in MODELS:
            raise ValueError(f"actuator model must be one of {MODELS}, got {model!r}")
        if model != "ideal" and rotor is None:
            raise ValueError(f"actuator model {model!r} needs rotor parameters")
        self.model = model
        self.rotor = rotor
        self.config_paths = tuple(Path(p) for p in config_paths)

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
        T, Q, _ = bemt_thrust(omega_cmd, airspeeds, self.rotor)
        return body_wrench(T, Q, self.rotor)


def load_actuator(vehicle_cfg: dict, vehicle_path: str | Path) -> RotorActuator:
    """Actuator from a vehicle config's [rotor] table; "ideal" when the table is absent.

    [rotor] model = "ideal" | "bemt", rotor_config = path relative to the vehicle file.
    """
    section = vehicle_cfg.get("rotor")
    if section is None:
        return RotorActuator("ideal")
    model = str(section.get("model", ""))
    if model == "ideal":
        return RotorActuator("ideal")
    if "rotor_config" not in section:
        raise ValueError(f"[rotor] model = {model!r} needs rotor_config")
    rotor_path = (Path(vehicle_path).resolve().parent / section["rotor_config"]).resolve()
    return RotorActuator(model, load_rotor_params(rotor_path), (rotor_path,))
