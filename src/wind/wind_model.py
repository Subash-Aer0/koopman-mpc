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
    def __init__(self, mean, cov, rng=None):
        raise NotImplementedError

    def sample(self, t, position):
        raise NotImplementedError


class LESWind:
    def __init__(self, path):
        raise NotImplementedError

    def sample(self, t, position):
        raise NotImplementedError
