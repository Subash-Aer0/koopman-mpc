"""Monomial lifting ψ(x) = (x, p_{n+1}(x), ..., p_N(x)), total degree ≤ p.

N = (n + p)! / (n! p!); n = 9, p = 2 gives N = 55 (Sec. IV-B).
"""
from itertools import combinations_with_replacement
from math import comb

import numpy as np


class MonomialLifting:
    """All monomials of the state with total degree ≤ p, the state itself first.

    Contract used by scripts/02_identify_models.py:
        ψ(x)[:n] == x, so C = [I_n 0] recovers the state from the lifted state;
        the remaining N − n entries are the other monomials, the constant 1
        included, in a fixed order.

    Order: x_1..x_n, then 1, then degree 2, 3, ..., p, each degree in
    lexicographic order of variable indices (x_1², x_1 x_2, ..., x_n²).
    """

    def __init__(self, state_dim, degree):
        """
        Args:
            state_dim: n, dimension of the (standardized) error state.
            degree: p, maximum total degree, ≥ 1.
        """
        if not (isinstance(state_dim, (int, np.integer)) and state_dim >= 1):
            raise ValueError(f"state_dim must be a positive integer, got {state_dim!r}")
        if not (isinstance(degree, (int, np.integer)) and degree >= 1):
            raise ValueError(f"degree must be a positive integer, got {degree!r}")
        self.state_dim = int(state_dim)
        self.degree = int(degree)

        rows = [np.eye(self.state_dim, dtype=int), np.zeros((1, self.state_dim), dtype=int)]
        for d in range(2, self.degree + 1):
            for indices in combinations_with_replacement(range(self.state_dim), d):
                rows.append(np.bincount(indices, minlength=self.state_dim)[None, :])
        # exponents[j, i] is the power of x_i in ψ_j.
        self.exponents = np.vstack(rows)
        self.exponents.setflags(write=False)

    @property
    def lifted_dim(self):
        """N = (n + p)! / (n! p!)."""
        return comb(self.state_dim + self.degree, self.degree)

    @property
    def constant_index(self):
        """Index j of the constant monomial, ψ_j(x) = 1; it follows the n state entries."""
        return self.state_dim

    def __call__(self, x):
        """x of shape (K, n) -> ψ(x) of shape (K, N); one sample per row."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.state_dim:
            raise ValueError(f"x must have shape (K, {self.state_dim}), got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains NaN or inf")
        # ψ_j(x) = Π_i x_i^exponents[j, i]; broadcast (K, 1, n) ** (N, n) -> (K, N, n).
        return np.prod(x[:, None, :] ** self.exponents[None, :, :], axis=2)
