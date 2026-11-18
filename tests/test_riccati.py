"""Time-varying Riccati recursion: DARE limit, periodic limit, equivalence with the condensed QP."""
import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from controllers.qp import condense_ltv
from controllers.riccati import time_varying_riccati

DT = 0.1


def double_integrator():
    A = np.array([[1.0, DT], [0.0, 1.0]])
    B = np.array([[0.5 * DT**2], [DT]])
    return A, B


def periodic_model(period: int, steps: int):
    """Double integrator whose input gain oscillates with the given period (in samples)."""
    A, B = double_integrator()
    gains = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(steps) / period)
    return np.repeat(A[None], steps, 0), B[None] * gains[:, None, None]


def test_constant_model_converges_to_the_dare():
    A, B = double_integrator()
    Q, R = np.diag([10.0, 1.0]), np.array([[0.1]])
    steps = 400
    P, K = time_varying_riccati(np.repeat(A[None], steps, 0), np.repeat(B[None], steps, 0),
                                Q, R, np.zeros((2, 2)))
    P_dare = solve_discrete_are(A, B, Q, R)
    K_dare = -np.linalg.solve(R + B.T @ P_dare @ B, B.T @ P_dare @ A)
    np.testing.assert_allclose(P[0], P_dare, rtol=1e-8)
    np.testing.assert_allclose(K[0], K_dare, rtol=1e-8)
    assert P.shape == (steps + 1, 2, 2) and K.shape == (steps, 1, 2)


def test_periodic_model_converges_to_a_periodic_solution_independent_of_the_end_weight():
    period, steps = 20, 400
    As, Bs = periodic_model(period, steps)
    Q, R = np.diag([10.0, 1.0]), np.array([[0.1]])
    P_a, _ = time_varying_riccati(As, Bs, Q, R, np.zeros((2, 2)))
    P_b, _ = time_varying_riccati(As, Bs, Q, R, 1e3 * np.eye(2))
    for k in range(period):
        np.testing.assert_allclose(P_a[k], P_a[k + period], rtol=1e-8)   # periodic
        np.testing.assert_allclose(P_a[k], P_b[k], rtol=1e-8)            # end weight forgotten
    # And it genuinely varies over the period (otherwise the test would be trivial).
    assert not np.allclose(P_a[0], P_a[period // 4], rtol=1e-3)


def test_cost_to_go_is_symmetric_positive_definite():
    As, Bs = periodic_model(20, 60)
    P, _ = time_varying_riccati(As, Bs, np.diag([10.0, 1.0]), np.array([[0.1]]), np.eye(2))
    for P_k in P:
        np.testing.assert_allclose(P_k, P_k.T, atol=0.0)
        assert np.linalg.eigvalsh(P_k).min() > 0


def test_unconstrained_ltv_mpc_with_riccati_terminal_weight_is_tv_lqr():
    # The LTV counterpart of test_qp.test_dare_terminal_weight_reproduces_lqr: with the
    # terminal weight P_N = P_{k+N}, the first MPC move is the time-varying LQR gain.
    period, horizon, steps = 20, 8, 200
    As, Bs = periodic_model(period, steps)
    Q, R = np.diag([10.0, 1.0]), np.array([[0.1]])
    P, K = time_varying_riccati(As, Bs, Q, R, np.zeros((2, 2)))
    x0 = np.array([1.0, -0.5])
    for k in (0, 5, 13):
        window = slice(k, k + horizon)
        qp = condense_ltv(As[window], Bs[window], np.zeros((horizon, 2, 1)), np.eye(2), Q, R,
                          P=P[k + horizon])
        V = -np.linalg.solve(qp.H, qp.linear_term(x0, np.zeros(1)))
        np.testing.assert_allclose(V[:1], K[k] @ x0, rtol=1e-8)


def test_bad_inputs_are_rejected():
    As, Bs = periodic_model(20, 5)
    Q, R = np.eye(2), np.eye(1)
    with pytest.raises(ValueError, match="Bs"):
        time_varying_riccati(As, Bs[:3], Q, R, Q)
    with pytest.raises(ValueError, match="R must be positive definite"):
        time_varying_riccati(As, Bs, Q, np.zeros((1, 1)), Q)
    with pytest.raises(ValueError, match="Q must be symmetric"):
        time_varying_riccati(As, Bs, np.array([[1.0, 1.0], [0.0, 1.0]]), R, Q)
    with pytest.raises(ValueError, match="finite"):
        time_varying_riccati(As * np.nan, Bs, Q, R, Q)
