"""Error metrics: RMSE (Tables I, III), Euclidean norm (Table II), MSE (Fig. 4)."""
import numpy as np


def rmse(err, axis=0) -> np.ndarray:
    """Root-mean-square of err along axis, in err's units."""
    err = np.asarray(err, dtype=float)
    if err.size == 0 or not np.all(np.isfinite(err)):
        raise ValueError("err must be non-empty and finite")
    return np.sqrt(np.mean(err**2, axis=axis))


def euclidean_norm_error(err):
    raise NotImplementedError


def percent_reduction(baseline, value):
    raise NotImplementedError
