"""Wind velocity v_w = v̄_w + ṽ_w.

- ConstantWind: mean only, ṽ_w = 0.
- GaussianWind: mean + zero-mean Gaussian turbulence, ṽ_w ~ N(0, Σvw).
- LESWind: replays a Large-Eddy-Simulation wind field [26] from data/wind/.

All models share sample(t, position) -> v_w: t in s, position in the NED
world frame in m, v_w in the NED world frame in m/s, shape (3,). Each model
owns its own time correlation; the simulator queries it every integration step.
"""
import numpy as np

from dynamics._validation import as_finite_vector


class ConstantWind:
    """Steady, uniform wind."""

    def __init__(self, velocity):
        """
        Args:
            velocity: v̄_w, NED world frame, m/s, shape (3,).
        """
        self.velocity = as_finite_vector("velocity", velocity, 3)
        self.velocity.setflags(write=False)

    def sample(self, t, position) -> np.ndarray:
        return self.velocity.copy()


class GaussianWind:
    """Uniform wind v̄_w + ṽ_w, ṽ_w ~ N(0, Σvw) redrawn every hold_period and held.

    Stand-in for the LES field of [26] (Sec. IV-A) until data/wind/ has one.
    With hold_period equal to the 10 Hz log period, each logged v_w,k is the
    wind acting over the whole interval [t_k, t_k+1), which is what Model 1,
    Eq. (5), assumes. Draws are made lazily in time order, so a run is
    reproducible from the rng seed alone.
    """

    def __init__(self, mean, cov, hold_period: float, rng: np.random.Generator):
        """
        Args:
            mean: v̄_w, NED world frame, m/s, shape (3,).
            cov: Σvw, (m/s)², shape (3, 3), symmetric positive definite.
            hold_period: Time each draw is held, s.
            rng: Source of the draws; owned by this object afterwards.
        """
        self.mean = as_finite_vector("mean", mean, 3)
        cov = np.asarray(cov, dtype=float)
        if cov.shape != (3, 3) or not np.all(np.isfinite(cov)):
            raise ValueError(f"cov must be a finite 3x3 matrix, got shape {cov.shape}")
        if not np.allclose(cov, cov.T):
            raise ValueError("cov must be symmetric")
        try:
            self._cov_factor = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as err:
            raise ValueError("cov must be positive definite") from err
        if not (np.isfinite(hold_period) and hold_period > 0):
            raise ValueError(f"hold_period must be positive and finite, got {hold_period}")
        self.hold_period = float(hold_period)
        self._rng = rng
        self._index = -1
        self._velocity = self.mean.copy()

    def sample(self, t, position) -> np.ndarray:
        # Small tolerance so t = k · hold_period computed in floating point lands in hold k.
        index = int(np.floor(t / self.hold_period + 1e-9))
        if index < self._index:
            raise ValueError(f"GaussianWind is sampled in time order; got t = {t} s "
                             f"after hold {self._index}")
        while self._index < index:  # one draw per hold, even for skipped holds
            self._velocity = self.mean + self._cov_factor @ self._rng.standard_normal(3)
            self._index += 1
        return self._velocity.copy()


class LESWind:
    def __init__(self, path):
        raise NotImplementedError

    def sample(self, t, position):
        raise NotImplementedError
