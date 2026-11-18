"""Monomial lifting ψ(x) = (x, p_{n+1}(x), ..., p_N(x)), total degree ≤ p.

N = (n + p)! / (n! p!); n = 9, p = 2 gives N = 55 (Sec. IV-B).
"""


class MonomialLifting:
    def __init__(self, state_dim, degree):
        raise NotImplementedError

    @property
    def lifted_dim(self):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError
