"""Rotor-speed dynamics of an open-loop (voltage-driven) brushless motor under aerodynamic load.

Per rotor, a DC-motor model with the supply voltage set by feed-forward from the
commanded speed (no speed feedback in the ESC):

    J_r Ω̇ = k_m (Ω_cmd − Ω) + k_Q Ω_cmd² − Q_aero          (1)

    J_r      rotor + propeller inertia about the spin axis, kg·m²
    k_m      motor stiffness K_t K_e / R, N·m·s/rad; τ_m = J_r / k_m is the time constant
    k_Q Ω_cmd²  feed-forward torque the nominal (quadratic) rotor model says is needed
    Q_aero   aerodynamic torque the rotor actually experiences (BEMT, dynamics/rotor.py)

Calibration: in still-air hover Q_aero = k_Q Ω², so Ω = Ω_cmd is the equilibrium of
(1): the motor, like the BEMT rotor, is exact in calm hover and wrong only in airflow.
In airflow, a larger aerodynamic torque slows the rotor:
    Ω_ss = Ω_cmd + (k_Q Ω_cmd² − Q_aero) / k_m.
A speed-controlled ESC would remove this droop (Ω → Ω_cmd), and with it the residual
thrust f_res of the CCTA paper's Model 2, which is estimated from the measured Ω.

Integration: Q_aero is held over the simulation step (as the wrench is), so (1) is
linear in Ω and is stepped exactly:
    Ω⁺ = Ω_ss + (Ω − Ω_ss) exp(−Δt / τ_m).

Units: Ω in rad/s, torque in N·m, time in s. Signs: Ω ≥ 0 (rotor speeds, not directions).
"""
from dataclasses import dataclass

import numpy as np

MODELS = ("ideal", "first_order")


@dataclass(frozen=True)
class MotorParams:
    """Motor parameters of Eq. (1), the same for every rotor."""

    time_constant: float     # τ_m, s
    rotor_inertia: float     # J_r, kg·m²

    def __post_init__(self) -> None:
        for name in ("time_constant", "rotor_inertia"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be positive and finite, got {value}")

    @property
    def stiffness(self) -> float:
        """k_m = J_r / τ_m, N·m·s/rad."""
        return self.rotor_inertia / self.time_constant


def steady_speed(omega_cmd, aero_torque, torque_coefficient: float,
                 params: MotorParams) -> np.ndarray:
    """Ω_ss of Eq. (1) for a held aerodynamic torque; rad/s, clipped at 0."""
    omega_cmd = np.asarray(omega_cmd, dtype=float)
    aero_torque = np.asarray(aero_torque, dtype=float)
    return np.maximum(
        omega_cmd + (torque_coefficient * omega_cmd**2 - aero_torque) / params.stiffness, 0.0)


def motor_step(omega, omega_cmd, aero_torque, torque_coefficient: float,
               params: MotorParams, dt: float) -> np.ndarray:
    """Exact step of Eq. (1) over dt with Q_aero held.

    Args:
        omega: Current rotor speeds Ω (n,), rad/s, ≥ 0.
        omega_cmd: Commanded speeds from the mixer (n,), rad/s, ≥ 0.
        aero_torque: Aerodynamic torque per rotor at the current Ω (n,), N·m.
        torque_coefficient: k_Q of the nominal model, N·m·s².
        params: Motor parameters.
        dt: Step, s, > 0.

    Returns:
        Ω after dt (n,), rad/s.
    """
    omega = np.asarray(omega, dtype=float)
    omega_cmd = np.asarray(omega_cmd, dtype=float)
    aero_torque = np.asarray(aero_torque, dtype=float)
    if not (omega.shape == omega_cmd.shape == aero_torque.shape):
        raise ValueError(f"omega, omega_cmd and aero_torque must share a shape, got "
                         f"{omega.shape}, {omega_cmd.shape}, {aero_torque.shape}")
    for name, value in (("omega", omega), ("omega_cmd", omega_cmd),
                        ("aero_torque", aero_torque)):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains NaN or inf")
    if np.any(omega < 0) or np.any(omega_cmd < 0):
        raise ValueError("rotor speeds must be non-negative")
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"dt must be positive and finite, got {dt}")
    target = steady_speed(omega_cmd, aero_torque, torque_coefficient, params)
    return target + (omega - target) * np.exp(-dt / params.time_constant)


def load_motor_params(section: dict | None) -> MotorParams | None:
    """MotorParams from a rotor config's [motor] table; None for model = "ideal" or no table."""
    if section is None:
        return None
    model = str(section.get("model", ""))
    if model not in MODELS:
        raise ValueError(f"[motor] model must be one of {MODELS}, got {model!r}")
    if model == "ideal":
        return None
    missing = [k for k in ("time_constant", "rotor_inertia") if k not in section]
    if missing:
        raise ValueError(f"[motor] model = 'first_order' needs {missing}")
    return MotorParams(time_constant=float(section["time_constant"]),
                       rotor_inertia=float(section["rotor_inertia"]))
