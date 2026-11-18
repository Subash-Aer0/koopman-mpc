"""Quadratic body drag f_D = R D v_rb ||v_rb||, with v_rb = Rᵀ v_r (Sec. II-A)."""
import numpy as np

from ._validation import as_finite_vector


def drag_force(rotation: np.ndarray, air_relative_velocity, drag_coeffs) -> np.ndarray:
    """Body drag in the world frame, N.

    Args:
        rotation: R ∈ SO(3), body (FRD) -> world (NED).
        air_relative_velocity: v_r = v − v_w, world frame, m/s, shape (3,).
        drag_coeffs: Diagonal of D, body axes, N·s²/m², shape (3,).

    Returns:
        f_D, world frame, N. Points along v_r; the dynamics subtract it (Eq. 3).
    """
    v_r = as_finite_vector("air_relative_velocity", air_relative_velocity, 3)
    v_rb = rotation.T @ v_r
    return rotation @ (np.asarray(drag_coeffs) * v_rb * np.linalg.norm(v_rb))
