"""Physical parameters of the quadcopter, rotors and drag model."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class QuadParams:
    mass: float = 1.0                                   # TODO
    inertia: np.ndarray = field(default_factory=lambda: np.diag([0.01, 0.01, 0.02]))  # TODO
    arm_length: float = 0.2                             # TODO
    gravity: float = 9.81
    drag_coeffs: np.ndarray = field(default_factory=lambda: np.diag([0.0, 0.0, 0.0]))  # D, TODO
    k_thrust: float = 0.0                               # quadratic rotor model, TODO
    k_torque: float = 0.0                               # TODO
