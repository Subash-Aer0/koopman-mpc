"""Monomial lifting with sines and cosines of the attitude angles.

    ψ(s) = (s, 1, t, monomials of degree 2..p in b),   b = (s[nonlinear], t)
    t    = (sin a_1, cos a_1 − 1, sin a_2, cos a_2 − 1, ...),   a_i = σ_i s[trig_i]  (rad)

Why trig features: the thrust direction R(η) e₃ and the body-frame airspeed
Rᵀ(v − v_w) are built from sines and cosines of the Euler angles, and their
products with velocity (degree 2 in b) are the terms of drag and thrust.

Design choices (each avoids a near-singular regressor):
- Angles are in radians: the lifting state s is scaled, s = (η − η*)/σ, so the
  trig features need σ (trig_scale) to recover a = η − η*. With the nominal
  trim (level, yaw 0) a is the attitude itself.
- cos a − 1 instead of cos a: cos a ≈ 1 near level and would duplicate the
  constant feature.
- Monomials with (cos a − 1)² or higher are left out: sin² a + (cos a − 1)²
  = −2 (cos a − 1) is an exact linear dependence on a feature already present.
- Angles listed in trig_indices should not also be in nonlinear_indices, or
  their raw monomials (a², a·v) nearly duplicate the trig ones.

Same contract as SelectiveMonomialLifting: ψ(s)[:n] == s (so C = [diag σ 0]
recovers the state) and `constant_index` locates ψ ≡ 1. With no trig_indices
the features and their order equal SelectiveMonomialLifting's.
"""
from itertools import combinations_with_replacement

import numpy as np


class TrigMonomialLifting:
    """s (K, n) -> ψ (K, N): the state, a constant, trig features, then monomials."""

    def __init__(self, state_dim: int, degree: int, nonlinear_indices, trig_indices=(),
                 trig_scale=None):
        """
        Args:
            state_dim: n, dimension of the (scaled) state.
            degree: p ≥ 1, maximum total degree of the monomials in b.
            nonlinear_indices: Distinct state indices whose raw values enter b.
            trig_indices: Distinct state indices of angles whose sin and cos − 1 enter b.
            trig_scale: σ (rad per unit of s) for each trig index, > 0; required
                when trig_indices is non-empty.
        """
        if not (isinstance(state_dim, (int, np.integer)) and state_dim >= 1):
            raise ValueError(f"state_dim must be a positive integer, got {state_dim!r}")
        if not (isinstance(degree, (int, np.integer)) and degree >= 1):
            raise ValueError(f"degree must be a positive integer, got {degree!r}")
        nonlinear = [int(i) for i in nonlinear_indices]
        trig = [int(i) for i in trig_indices]
        for name, indices in (("nonlinear_indices", nonlinear), ("trig_indices", trig)):
            if len(set(indices)) != len(indices) or any(not 0 <= i < state_dim for i in indices):
                raise ValueError(f"{name} must be distinct and in [0, {state_dim}), got {indices}")
        if set(nonlinear) & set(trig):
            raise ValueError(f"an angle may be raw or trig, not both: {sorted(set(nonlinear) & set(trig))}")
        if trig:
            scale = np.asarray(trig_scale, dtype=float)
            if scale.shape != (len(trig),) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError(f"trig_scale must be {len(trig)} positive finite values, "
                                 f"got {trig_scale}")
        else:
            scale = np.zeros(0)
        self.state_dim = int(state_dim)
        self.degree = int(degree)
        self.nonlinear_indices = tuple(sorted(nonlinear))
        self.trig_indices = tuple(trig)
        self.trig_scale = scale
        self.trig_scale.setflags(write=False)

        # Base variables b: raw nonlinear states, then (sin, cos − 1) per trig angle.
        n_raw, n_trig = len(self.nonlinear_indices), 2 * len(self.trig_indices)
        self._is_cos = np.array([False] * n_raw + [False, True] * len(self.trig_indices))
        rows = [np.eye(n_raw + n_trig, dtype=int)[n_raw:]]          # trig features, degree 1
        for d in range(2, self.degree + 1):
            for combo in combinations_with_replacement(range(n_raw + n_trig), d):
                powers = np.bincount(combo, minlength=n_raw + n_trig)
                if np.any(powers[self._is_cos] >= 2):
                    continue                                       # sin² + cos² = 1
                rows.append(powers[None, :])
        # exponents[j, k] is the power of b_k in the j-th feature after (s, 1).
        self.exponents = np.vstack(rows) if rows else np.zeros((0, n_raw + n_trig), dtype=int)
        self.exponents.setflags(write=False)

    @property
    def lifted_dim(self) -> int:
        return self.state_dim + 1 + self.exponents.shape[0]

    @property
    def constant_index(self) -> int:
        """Index j of ψ_j ≡ 1; it follows the n state entries."""
        return self.state_dim

    def _base(self, s: np.ndarray) -> np.ndarray:
        angles = s[:, list(self.trig_indices)] * self.trig_scale         # rad
        trig = np.empty((len(s), 2 * len(self.trig_indices)))
        trig[:, 0::2] = np.sin(angles)
        trig[:, 1::2] = np.cos(angles) - 1.0
        return np.hstack([s[:, list(self.nonlinear_indices)], trig])

    def base_names(self, state_names) -> list[str]:
        names = [state_names[i] for i in self.nonlinear_indices]
        for i in self.trig_indices:
            names += [f"sin({state_names[i]})", f"(cos({state_names[i]})-1)"]
        return names

    def feature_names(self, state_names) -> list[str]:
        """Readable name of each ψ_j, e.g. 'v_x*sin(roll)', for spectra and reports."""
        if len(state_names) != self.state_dim:
            raise ValueError(f"need {self.state_dim} state names, got {len(state_names)}")
        base = self.base_names(state_names)
        names = list(state_names) + ["1"]
        for row in self.exponents:
            factors = []
            for k, power in enumerate(row):
                factors += [base[k]] * int(power)
            names.append("*".join(factors))
        return names

    def __call__(self, s) -> np.ndarray:
        """s (K, n) -> ψ (K, N); one sample per row."""
        s = np.asarray(s, dtype=float)
        if s.ndim != 2 or s.shape[1] != self.state_dim:
            raise ValueError(f"s must have shape (K, {self.state_dim}), got {s.shape}")
        if not np.all(np.isfinite(s)):
            raise ValueError("s contains NaN or inf")
        b = self._base(s)
        extra = np.prod(b[:, None, :] ** self.exponents[None, :, :], axis=2)
        return np.hstack([s, np.ones((len(s), 1)), extra])
