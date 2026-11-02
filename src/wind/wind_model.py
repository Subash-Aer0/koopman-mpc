"""Wind velocity v_w = v̄_w + ṽ_w.

- GaussianWind: mean + zero-mean Gaussian turbulence, ṽ_w ~ N(0, Σvw).
- LESWind: replays a Large-Eddy-Simulation wind field [26] from data/wind/.
"""


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
