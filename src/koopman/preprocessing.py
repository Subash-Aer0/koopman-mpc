"""Data conditioning for EDMD: state standardization and regressor diagnostics.

Standardizing e_x before lifting puts position (m), velocity (m/s) and attitude
(rad) errors on one scale, so monomials like φ² are not ~100x smaller than
p_x², the regressor is well conditioned and the l1 penalty in Eq. (8) treats
features evenly. Inputs u_d and v_w are left in physical units, so B and D
act on the same quantities the MPC commands and measures.
"""
from dataclasses import dataclass

import numpy as np

# Channels with std below this are treated as unexcited rather than divided by ~0.
_MIN_STD = 1e-12


@dataclass(frozen=True)
class Standardizer:
    """z = (e − mean) / std per channel; fit on training data only.

    Attributes:
        mean: Per-channel mean, shape (n,), units of e.
        std: Per-channel standard deviation, shape (n,), units of e.
    """

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.mean.shape != self.std.shape:
            raise ValueError(f"mean and std must be equal 1-D shapes, got "
                             f"{self.mean.shape} and {self.std.shape}")
        if not (np.all(np.isfinite(self.mean)) and np.all(np.isfinite(self.std))):
            raise ValueError("mean or std contains NaN or inf")
        if np.any(self.std <= _MIN_STD):
            raise ValueError(f"std must be positive in every channel, got {self.std}")

    @classmethod
    def fit(cls, data) -> "Standardizer":
        """Fit to samples along all leading axes, e.g. e_x of shape (E, T, n).

        Raises:
            ValueError: If a channel is constant; it carries no information to
                identify and would make z = (e − mean) / 0.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim < 2:
            raise ValueError(f"data must have shape (..., n), got {data.shape}")
        flat = data.reshape(-1, data.shape[-1])
        if not np.all(np.isfinite(flat)):
            raise ValueError("data contains NaN or inf")
        std = flat.std(axis=0)
        if np.any(std <= _MIN_STD):
            raise ValueError(f"channels {np.flatnonzero(std <= _MIN_STD).tolist()} are constant "
                             "in the data and cannot be standardized")
        return cls(mean=flat.mean(axis=0), std=std)

    def transform(self, e) -> np.ndarray:
        """e (..., n) in physical units -> z (..., n), dimensionless."""
        return (np.asarray(e, dtype=float) - self.mean) / self.std

    def inverse(self, z) -> np.ndarray:
        """z (..., n), dimensionless -> e (..., n) in physical units."""
        return np.asarray(z, dtype=float) * self.std + self.mean


def regressor_diagnostics(gamma1, expected_rank: int) -> dict:
    """Rank and condition number of the EDMD regressor Γ1, columns α1[k] (Eq. 7).

    [A B D] is unique only if Γ1 has full row rank, N + m + d (persistence of
    excitation). The condition number says how sensitive the fit is to noise.

    Args:
        gamma1: Γ1, shape (N + m + d, K), one snapshot per column.
        expected_rank: N + m + d.

    Returns:
        {"rows", "columns", "rank", "condition_number"}.

    Raises:
        ValueError: If there are fewer snapshots than rows or the rank is
            below expected_rank; the fit would not be unique.
    """
    gamma1 = np.asarray(gamma1, dtype=float)
    if gamma1.ndim != 2 or gamma1.shape[0] != expected_rank:
        raise ValueError(f"gamma1 must have {expected_rank} rows, got shape {gamma1.shape}")
    if not np.all(np.isfinite(gamma1)):
        raise ValueError("gamma1 contains NaN or inf")
    rows, columns = gamma1.shape
    if columns < rows:
        raise ValueError(f"{columns} snapshots cannot identify {rows} regressors")
    singular_values = np.linalg.svd(gamma1, compute_uv=False)
    rank = int(np.linalg.matrix_rank(gamma1))
    if rank < expected_rank:
        raise ValueError(f"Γ1 has rank {rank} < {expected_rank}: some lifted feature, input "
                         "or wind channel is not independently excited by the data")
    return {
        "rows": rows,
        "columns": columns,
        "rank": rank,
        "condition_number": float(singular_values[0] / singular_values[-1]),
    }
