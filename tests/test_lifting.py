"""Tests for lifting: dimension, state-first contract, monomial values."""
from math import comb

import numpy as np
import pytest

from koopman.lifting import MonomialLifting


@pytest.mark.parametrize("n, p", [(9, 2), (9, 3), (2, 1), (3, 4)])
def test_lifted_dim_matches_binomial_and_exponents(n, p):
    lifting = MonomialLifting(n, p)
    assert lifting.lifted_dim == comb(n + p, p)
    assert lifting.exponents.shape == (lifting.lifted_dim, n)
    # Every monomial of degree ≤ p appears exactly once.
    assert len({tuple(row) for row in lifting.exponents}) == lifting.lifted_dim
    assert lifting.exponents.sum(axis=1).max() == p


def test_paper_dimension_is_55():
    assert MonomialLifting(9, 2).lifted_dim == 55


def test_state_first_then_constant():
    x = np.random.default_rng(0).standard_normal((6, 9))
    psi = MonomialLifting(9, 2)(x)
    np.testing.assert_array_equal(psi[:, :9], x)
    np.testing.assert_array_equal(psi[:, 9], 1.0)
    assert MonomialLifting(9, 2).constant_index == 9


def test_degree_two_values_by_hand():
    # n = 2, p = 2: ψ = [x1, x2, 1, x1², x1 x2, x2²].
    psi = MonomialLifting(2, 2)(np.array([[2.0, -3.0]]))
    np.testing.assert_array_equal(psi, [[2.0, -3.0, 1.0, 4.0, -6.0, 9.0]])


def test_zero_state_lifts_to_unit_constant_only():
    psi = MonomialLifting(4, 3)(np.zeros((1, 4)))
    expected = np.zeros(MonomialLifting(4, 3).lifted_dim)
    expected[4] = 1.0
    np.testing.assert_array_equal(psi[0], expected)


def test_rejects_wrong_shape_and_nan():
    lifting = MonomialLifting(3, 2)
    with pytest.raises(ValueError, match=r"\(K, 3\)"):
        lifting(np.zeros(3))
    with pytest.raises(ValueError, match="NaN"):
        lifting(np.array([[0.0, np.nan, 1.0]]))


def test_rejects_bad_degree():
    with pytest.raises(ValueError, match="degree"):
        MonomialLifting(3, 0)
