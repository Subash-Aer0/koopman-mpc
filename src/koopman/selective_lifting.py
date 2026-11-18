"""Monomial lifting with nonlinear features in selected state variables only.

ψ(x) = (x, 1, monomials of degree 2..p in x[nonlinear_indices]).

Why selective: the quadrotor's dynamics do not depend on position (translation
invariance), so position should enter the model linearly; monomials in
position could only fit noise. The nonlinearity lives in velocity (drag) and
attitude (thrust direction). Choosing the variables keeps the dictionary small
(about 30 features instead of 55), which needs less data and adds fewer
spurious near-unit modes (docs/koopman_mpc_study_guide.md §10.4).

Same contract as MonomialLifting: ψ(x)[:n] == x, so C = [I_n 0] recovers the
state, and `constant_index` locates the constant feature for fit_edmd.
"""
from itertools import combinations_with_replacement

import numpy as np


class SelectiveMonomialLifting:
    """x (K, n) -> ψ (K, N): the state, a constant, then higher-degree monomials.

    Order: x_1..x_n, then 1, then degree 2, 3, ..., p over the selected
    variables, each degree in lexicographic order of their indices.
    """

    def __init__(self, state_dim: int, degree: int, nonlinear_indices):
        """
        Args:
            state_dim: n, dimension of the (scaled) state.
            degree: p ≥ 1, maximum total degree of the extra monomials.
            nonlinear_indices: Distinct state indices the monomials are built from.
        """
        if not (isinstance(state_dim, (int, np.integer)) and state_dim >= 1):
            raise ValueError(f"state_dim must be a positive integer, got {state_dim!r}")
        if not (isinstance(degree, (int, np.integer)) and degree >= 1):
            raise ValueError(f"degree must be a positive integer, got {degree!r}")
        indices = [int(i) for i in nonlinear_indices]
        if len(set(indices)) != len(indices) or any(not 0 <= i < state_dim for i in indices):
            raise ValueError(f"nonlinear_indices must be distinct and in [0, {state_dim}), "
                             f"got {list(nonlinear_indices)}")
        self.state_dim = int(state_dim)
        self.degree = int(degree)
        self.nonlinear_indices = tuple(sorted(indices))

        rows = [np.eye(self.state_dim, dtype=int), np.zeros((1, self.state_dim), dtype=int)]
        for d in range(2, self.degree + 1):
            for combo in combinations_with_replacement(self.nonlinear_indices, d):
                rows.append(np.bincount(combo, minlength=self.state_dim)[None, :])
        # exponents[j, i] is the power of x_i in ψ_j.
        self.exponents = np.vstack(rows)
        self.exponents.setflags(write=False)

    @property
    def lifted_dim(self) -> int:
        return self.exponents.shape[0]

    @property
    def constant_index(self) -> int:
        """Index j of ψ_j ≡ 1; it follows the n state entries."""
        return self.state_dim

    def feature_names(self, state_names) -> list[str]:
        """Readable name of each ψ_j, e.g. 'v_x*roll', for spectra and reports."""
        if len(state_names) != self.state_dim:
            raise ValueError(f"need {self.state_dim} state names, got {len(state_names)}")
        names = []
        for row in self.exponents:
            if not row.any():
                names.append("1")
                continue
            factors = []
            for i, power in enumerate(row):
                factors += [state_names[i]] * int(power)
            names.append("*".join(factors))
        return names

    def __call__(self, x) -> np.ndarray:
        """x (K, n) -> ψ (K, N); one sample per row."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.state_dim:
            raise ValueError(f"x must have shape (K, {self.state_dim}), got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains NaN or inf")
        return np.prod(x[:, None, :] ** self.exponents[None, :, :], axis=2)
