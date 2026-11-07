"""Input checks shared by the public functions of the dynamics package."""
import numpy as np


def as_finite_vector(name: str, value, size: int) -> np.ndarray:
    """Return `value` as a float array of shape (size,), or raise ValueError."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf: {arr}")
    return arr
