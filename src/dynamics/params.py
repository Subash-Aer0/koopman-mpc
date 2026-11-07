"""Rigid-body and drag parameters of the quadcopter.

Values are loaded from a TOML config (see configs/quadrotor.toml) rather than
hard-coded, so each experiment is reproducible from its config file.
Rotor parameters live with the rotor model (rotor.py), not here.
"""
from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np


@dataclass(frozen=True)
class QuadParams:
    """Physical parameters used by the rigid-body dynamics.

    Attributes:
        mass: Vehicle mass m, kg.
        inertia: Moment of inertia J about the centre of mass, body FRD frame,
            kg·m², shape (3, 3), symmetric positive definite.
        drag_coeffs: Diagonal of the drag matrix D (Sec. II-A), body FRD axes,
            N·s²/m², shape (3,), non-negative.
        gravity: Gravitational acceleration magnitude g, m/s². Acts along +z
            of the NED world frame.
    """

    mass: float
    inertia: np.ndarray
    drag_coeffs: np.ndarray
    gravity: float

    def __post_init__(self) -> None:
        inertia = np.array(self.inertia, dtype=float)
        drag = np.array(self.drag_coeffs, dtype=float)

        if not (np.isfinite(self.mass) and self.mass > 0):
            raise ValueError(f"mass must be positive and finite, got {self.mass}")
        if not (np.isfinite(self.gravity) and self.gravity > 0):
            raise ValueError(f"gravity must be positive and finite, got {self.gravity}")
        if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
            raise ValueError(f"inertia must be a finite 3x3 matrix, got shape {inertia.shape}")
        if not np.allclose(inertia, inertia.T):
            raise ValueError("inertia must be symmetric")
        if np.any(np.linalg.eigvalsh(inertia) <= 0):
            raise ValueError("inertia must be positive definite")
        if drag.shape != (3,) or not np.all(np.isfinite(drag)) or np.any(drag < 0):
            raise ValueError(f"drag_coeffs must be 3 finite non-negative values, got {drag}")

        # Read-only copies so a shared QuadParams cannot be mutated mid-simulation.
        inertia.setflags(write=False)
        drag.setflags(write=False)
        object.__setattr__(self, "mass", float(self.mass))
        object.__setattr__(self, "gravity", float(self.gravity))
        object.__setattr__(self, "inertia", inertia)
        object.__setattr__(self, "drag_coeffs", drag)

    @property
    def gravity_vector(self) -> np.ndarray:
        """Gravity in the NED world frame, [0, 0, g], m/s²."""
        return np.array([0.0, 0.0, self.gravity])


def load_params(path: str | Path) -> QuadParams:
    """Load QuadParams from a TOML file laid out like configs/quadrotor.toml."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    return QuadParams(
        mass=cfg["vehicle"]["mass"],
        inertia=cfg["vehicle"]["inertia"],
        drag_coeffs=cfg["drag"]["coefficients"],
        gravity=cfg["environment"]["gravity"],
    )
