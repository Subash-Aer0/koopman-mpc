"""Rotor models (Sec. II-A).

- Quadratic rotor model: T = k_T Ω², Q = k_Q Ω². What the controller believes:
  the mixer inverts it to turn (collective thrust, body torque) into rotor
  speeds, and it estimates executed thrust f_r from measured Ω_R (Model 2).
- Blade Element Momentum Theory (BEMT): the thrust and torque a rotor really
  produces at speed Ω in its local airflow. Implemented here in the classical
  uniform-inflow form (Leishman, Principles of Helicopter Aerodynamics, 2nd ed.,
  Ch. 2-3 and 5), NOT the hybrid BEMT of Davoudi et al. [1] used in the CCTA
  paper (non-uniform inflow, rotor-speed adjustment). Rotor speed is an input:
  the command for an ideal motor (Ω_R = Ω_cmd), or the state of the optional
  motor model (dynamics/motor.py, wired in dynamics/actuator.py), which slows
  under aerodynamic torque like the paper's rotor-speed adjustment.

BEMT, per rotor (non-dimensional by the tip speed ΩR):
    μ   = V_edge / (ΩR)       advance ratio, airflow in the rotor disk plane
    λ_c = V_axial / (ΩR)      axial inflow, > 0 when air flows down through the
                              disk (climb relative to the air), body FRD: −v_z
    λ   = λ_c + λ_i           total inflow; λ_i induced, the unknown
  blade element (constant chord, linear twist θ(r) = θ₀ + θ_tw r, r = radius / R):
    C_T = (σ a / 2) [θ₀/3 (1 + 3μ²/2) + θ_tw/4 (1 + μ²) − λ/2]
  momentum (Glauert):
    λ_i = C_T / (2 √(μ² + λ²))
  torque (induced + climb power, profile power):
    C_Q = λ C_T + σ C_d0 / 8 (1 + 4.6 μ²)
  dimensional:  T = C_T ρ A (ΩR)²,  Q = C_Q ρ A (ΩR)² R,  A = π R²,  σ = N_b c / (π R)

Calibration: θ₀ and C_d0 are chosen so that in still-air hover BEMT gives
exactly k_T Ω² and k_Q Ω². Hover C_T and C_Q do not depend on Ω, so this holds
at every rotor speed: the nominal model is exact in calm hover and wrong only
in airflow, which is the mismatch under study.

Validity: momentum theory fails in the vortex ring state (moderate descent
through the rotor's own wake); bemt_thrust raises once the descent rate
reaches half the hover induced velocity v_h (−λ_c ≥ 0.5 λ_h). Reverse flow, tip losses, stall and rotor
drag (H-force) are not modelled.

Frames and units: body FRD, rotor thrust along −z_body; Ω in rad/s, T in N,
Q in N·m, airspeeds in m/s.
"""
from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np

N_ROTORS = 4
_NEWTON_TOL = 1e-12
_NEWTON_MAX_ITERATIONS = 50
_VORTEX_RING_ONSET = 0.5        # −V_axial / v_h at which momentum theory is abandoned
_PROFILE_POWER_FACTOR = 4.6     # C_Q profile term ∝ (1 + 4.6 μ²), Leishman Eq. 5.x


@dataclass(frozen=True)
class RotorParams:
    """Quadratic model, rotor layout and blade geometry (configs/rotor.toml)."""

    thrust_coefficient: float     # k_T, N·s²
    torque_coefficient: float     # k_Q, N·m·s²
    positions: np.ndarray         # r_i, body FRD, m, (4, 3)
    yaw_torque_signs: np.ndarray  # s_i ∈ {+1, −1}, (4,)
    radius: float                 # R, m
    chord: float                  # c, m
    blades: int                   # N_b
    twist: float                  # θ_tw, rad
    lift_slope: float             # a, 1/rad
    air_density: float            # ρ, kg/m³

    def __post_init__(self) -> None:
        for name in ("thrust_coefficient", "torque_coefficient", "radius", "chord",
                     "lift_slope", "air_density"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be positive and finite, got {value}")
        if not (isinstance(self.blades, (int, np.integer)) and self.blades >= 1):
            raise ValueError(f"blades must be a positive integer, got {self.blades!r}")
        if not np.isfinite(self.twist):
            raise ValueError(f"twist must be finite, got {self.twist}")
        positions = np.array(self.positions, dtype=float)
        signs = np.array(self.yaw_torque_signs, dtype=float)
        if positions.shape != (N_ROTORS, 3) or not np.all(np.isfinite(positions)):
            raise ValueError(f"positions must be a finite (4, 3) array, got {positions.shape}")
        if signs.shape != (N_ROTORS,) or not np.all(np.abs(signs) == 1.0):
            raise ValueError(f"yaw_torque_signs must be four ±1, got {self.yaw_torque_signs}")
        if np.linalg.matrix_rank(_allocation(positions, signs, 1.0)) < N_ROTORS:
            raise ValueError("rotor layout cannot produce independent thrust and torques")
        positions.setflags(write=False)
        signs.setflags(write=False)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "yaw_torque_signs", signs)
        if self.profile_drag <= 0:
            raise ValueError(f"k_Q is too small for this blade: calibrated C_d0 = "
                             f"{self.profile_drag:.4f} ≤ 0 (induced power alone exceeds k_Q)")

    # --- derived blade quantities -------------------------------------------------

    @property
    def disk_area(self) -> float:
        return np.pi * self.radius**2

    @property
    def solidity(self) -> float:
        """σ = N_b c / (π R)."""
        return self.blades * self.chord / (np.pi * self.radius)

    @property
    def hover_thrust_coeff(self) -> float:
        """C_T in still-air hover, = k_T / (ρ A R²), independent of Ω."""
        return self.thrust_coefficient / (self.air_density * self.disk_area * self.radius**2)

    @property
    def hover_inflow(self) -> float:
        """λ_h = √(C_T,h / 2)."""
        return np.sqrt(self.hover_thrust_coeff / 2.0)

    @property
    def root_pitch(self) -> float:
        """θ₀, rad, calibrated: blade-element C_T at hover inflow equals C_T,h."""
        c_t, lam = self.hover_thrust_coeff, self.hover_inflow
        return 3.0 * (2.0 * c_t / (self.solidity * self.lift_slope) - self.twist / 4.0
                      + lam / 2.0)

    @property
    def profile_drag(self) -> float:
        """C_d0, calibrated: hover C_Q = λ_h C_T,h + σ C_d0 / 8 equals k_Q / (ρ A R³)."""
        c_q = self.torque_coefficient / (self.air_density * self.disk_area * self.radius**3)
        return 8.0 * (c_q - self.hover_inflow * self.hover_thrust_coeff) / self.solidity


def load_rotor_params(path: str | Path) -> RotorParams:
    """Load RotorParams from a TOML file laid out like configs/rotor.toml."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    geometry, blade = cfg["geometry"], cfg["blade"]
    beta = np.radians(np.asarray(geometry["arm_angles_deg"], dtype=float))
    positions = geometry["arm_length"] * np.column_stack([np.cos(beta), np.sin(beta),
                                                          np.zeros_like(beta)])
    return RotorParams(
        thrust_coefficient=cfg["quadratic"]["thrust_coefficient"],
        torque_coefficient=cfg["quadratic"]["torque_coefficient"],
        positions=positions, yaw_torque_signs=geometry["yaw_torque_signs"],
        radius=blade["radius"], chord=blade["chord"], blades=blade["blades"],
        twist=blade["twist"], lift_slope=blade["lift_slope"],
        air_density=cfg["environment"]["air_density"],
    )


# --- quadratic model, mixer and body wrench ---------------------------------------

def _allocation(positions, yaw_signs, torque_per_thrust: float) -> np.ndarray:
    """M with [f_c, τ_x, τ_y, τ_z] = M T for rotor thrusts T (4,), thrust along −z_body.

    Force F_i = [0, 0, −T_i] at r_i = [x_i, y_i, 0] gives τ_i = r_i × F_i = [−y_i T_i, x_i T_i, 0];
    the reaction torque s_i Q_i = s_i (k_Q/k_T) T_i adds to τ_z.
    """
    return np.vstack([np.ones(N_ROTORS), -positions[:, 1], positions[:, 0],
                      torque_per_thrust * yaw_signs])


def body_wrench(thrusts, torques, params: RotorParams) -> tuple[np.ndarray, np.ndarray]:
    """Rotor thrusts T (4,) N and reaction torques Q (4,) N·m -> f_t (3,) N, τ_t (3,) N·m, body FRD."""
    T = np.asarray(thrusts, dtype=float)
    Q = np.asarray(torques, dtype=float)
    if T.shape != (N_ROTORS,) or Q.shape != (N_ROTORS,):
        raise ValueError(f"thrusts and torques must have shape (4,), got {T.shape}, {Q.shape}")
    r = params.positions
    f_t = np.array([0.0, 0.0, -np.sum(T)])
    tau_t = np.array([-r[:, 1] @ T, r[:, 0] @ T, params.yaw_torque_signs @ Q])
    return f_t, tau_t


def quadratic_thrust_torque(rotor_speeds, params: RotorParams) -> tuple[np.ndarray, np.ndarray]:
    """T = k_T Ω², Q = k_Q Ω² per rotor; Ω (4,) rad/s -> T (4,) N, Q (4,) N·m."""
    omega = np.asarray(rotor_speeds, dtype=float)
    if omega.shape != (N_ROTORS,) or not np.all(np.isfinite(omega)) or np.any(omega < 0):
        raise ValueError(f"rotor_speeds must be 4 finite non-negative values, got {omega}")
    return params.thrust_coefficient * omega**2, params.torque_coefficient * omega**2


def mixer(collective_thrust: float, body_torque, params: RotorParams) -> np.ndarray:
    """Invert the quadratic model: (f_c N, τ (3,) N·m body FRD) -> commanded Ω (4,) rad/s.

    Rotor thrusts that would be negative are clipped to 0 (a rotor cannot pull),
    so the requested wrench is then not met exactly.
    """
    tau = np.asarray(body_torque, dtype=float)
    if not np.isfinite(collective_thrust) or tau.shape != (3,) or not np.all(np.isfinite(tau)):
        raise ValueError(f"need a finite collective thrust and body torque (3,), got "
                         f"{collective_thrust}, {body_torque}")
    M = _allocation(params.positions, params.yaw_torque_signs,
                    params.torque_coefficient / params.thrust_coefficient)
    T = np.linalg.solve(M, np.concatenate([[collective_thrust], tau]))
    return np.sqrt(np.clip(T, 0.0, None) / params.thrust_coefficient)


def rotor_airspeeds(air_velocity_body, body_rate, params: RotorParams) -> np.ndarray:
    """Air-relative velocity of each rotor hub, body FRD, m/s, (4, 3): v_rb + ω × r_i."""
    v = np.asarray(air_velocity_body, dtype=float)
    w = np.asarray(body_rate, dtype=float)
    if v.shape != (3,) or w.shape != (3,) or not (np.all(np.isfinite(v)) and np.all(np.isfinite(w))):
        raise ValueError("air_velocity_body and body_rate must be finite (3,) vectors")
    return v + np.cross(w, params.positions)


# --- BEMT ------------------------------------------------------------------------

def bemt_inflow(advance_ratio, axial_inflow, params: RotorParams):
    """Solve blade element = momentum for the total inflow λ; vectorized over rotors.

    Args:
        advance_ratio: μ ≥ 0, shape (k,).
        axial_inflow: λ_c, shape (k,); > 0 for air flowing down through the disk.
        params: Blade parameters.

    Returns:
        (λ, C_T), each shape (k,).

    Raises:
        ValueError: If Newton's method does not converge, or the rotor is in the
            vortex ring state (−λ_c ≥ 0.5 λ_h), outside momentum theory.
    """
    mu = np.asarray(advance_ratio, dtype=float)
    lam_c = np.asarray(axial_inflow, dtype=float)
    k_lam = params.solidity * params.lift_slope / 4.0          # C_T = c0 − k_lam λ
    c0 = (params.solidity * params.lift_slope / 2.0) * (
        params.root_pitch / 3.0 * (1.0 + 1.5 * mu**2) + params.twist / 4.0 * (1.0 + mu**2))
    lam = lam_c + params.hover_inflow
    for _ in range(_NEWTON_MAX_ITERATIONS):
        s = np.sqrt(mu**2 + lam**2)
        c_t = c0 - k_lam * lam
        f = lam - lam_c - c_t / (2.0 * s)
        df = 1.0 + k_lam / (2.0 * s) + c_t * lam / (2.0 * s**3)
        step = f / df
        lam = lam - step
        if np.all(np.abs(step) < _NEWTON_TOL):
            break
    else:
        raise ValueError(f"BEMT inflow did not converge for μ = {mu}, λ_c = {lam_c}")
    c_t = c0 - k_lam * lam
    if np.any(-lam_c >= _VORTEX_RING_ONSET * params.hover_inflow):
        raise ValueError(f"vortex ring state: descent λ_c = {lam_c} vs hover inflow "
                         f"{params.hover_inflow:.4f}; momentum theory does not apply")
    return lam, c_t


def bemt_thrust(rotor_speeds_cmd, relative_airspeed_body, params: RotorParams):
    """Executed thrust, reaction torque and rotor speed of each rotor in its airflow.

    Args:
        rotor_speeds_cmd: Ω (4,), rad/s, > 0.
        relative_airspeed_body: Air-relative velocity of each rotor hub, body
            FRD, m/s, (4, 3) (rotor_airspeeds).
        params: Rotor parameters.

    Returns:
        T (4,) N, Q (4,) N·m, and the speeds they were evaluated at (4,) rad/s. Pass the
        command for an ideal motor, or the motor state Ω (dynamics/motor.py).
        A stopped rotor (Ω = 0, e.g. clipped by the mixer) gives T = Q = 0.
    """
    omega = np.asarray(rotor_speeds_cmd, dtype=float)
    v = np.asarray(relative_airspeed_body, dtype=float)
    if omega.shape != (N_ROTORS,) or not np.all(np.isfinite(omega)) or np.any(omega < 0):
        raise ValueError(f"rotor_speeds_cmd must be 4 finite non-negative values, got {omega}")
    if v.shape != (N_ROTORS, 3) or not np.all(np.isfinite(v)):
        raise ValueError(f"relative_airspeed_body must be a finite (4, 3) array, got {v.shape}")
    T, Q = np.zeros(N_ROTORS), np.zeros(N_ROTORS)
    spinning = omega > 0
    if np.any(spinning):
        tip_speed = omega[spinning] * params.radius
        vs = v[spinning]
        mu = np.linalg.norm(vs[:, :2], axis=1) / tip_speed
        lam_c = -vs[:, 2] / tip_speed          # moving up (−z) through the air: flow down the disk
        lam, c_t = bemt_inflow(mu, lam_c, params)
        c_q = lam * c_t + params.solidity * params.profile_drag / 8.0 * (
            1.0 + _PROFILE_POWER_FACTOR * mu**2)
        dynamic = params.air_density * params.disk_area * tip_speed**2
        T[spinning] = c_t * dynamic
        Q[spinning] = c_q * dynamic * params.radius
    return T, Q, omega.copy()
