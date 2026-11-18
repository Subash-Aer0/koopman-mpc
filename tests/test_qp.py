"""Condensed MPC QP: prediction vs rollout, cost, LQR equivalence, constraints, solver status.

Tests 1-4 and 6 of docs/mpc_guide.md §5.9.
"""
import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from controllers.qp import condense, condense_ltv, solve_qp
from koopman.models import KoopmanModel

DT = 0.1


def double_integrator():
    """1-D double integrator, x = [p, v], u = a, wind enters as an acceleration."""
    A = np.array([[1.0, DT], [0.0, 1.0]])
    B = np.array([[0.5 * DT**2], [DT]])
    D = B.copy()
    return A, B, D, np.eye(2)


def random_model(rng, N=6, m=2, d=3, n=4):
    """A random lifted model with spectral radius 0.9 and C = [I_n 0]."""
    A = rng.standard_normal((N, N))
    A *= 0.9 / np.abs(np.linalg.eigvals(A)).max()
    C = np.hstack([np.eye(n), np.zeros((n, N - n))])
    return A, rng.standard_normal((N, m)), rng.standard_normal((N, d)), C


def rollout_cost(A, B, D, C, Q, R, P, z0, V, v_w, horizon, r=None):
    """J of Eq. (15) without the constant z_0 term, by simulating the dynamics.

    r: output reference r_1..r_np, shape (np, n); default 0.
    """
    m = B.shape[1]
    r = np.zeros((horizon, C.shape[0])) if r is None else r
    z, J = z0, 0.0
    for k in range(horizon):
        nu = V[k * m:(k + 1) * m]
        J += nu @ R @ nu
        z = A @ z + B @ nu + D @ v_w
        W = P if k == horizon - 1 else Q
        e = C @ z - r[k]
        J += e @ W @ e
    return J


def test_prediction_matches_koopman_model_rollout():
    rng = np.random.default_rng(0)
    A, B, D, C = random_model(rng)
    model = KoopmanModel(A=A, B=B, D=D, C=C)
    horizon = 5
    qp = condense(A, B, D, C, np.eye(4), np.eye(2), horizon)
    z0, v_w = rng.standard_normal(6), rng.standard_normal(3)
    V = rng.standard_normal(horizon * 2)

    z, expected = z0, []
    for k in range(horizon):
        z = model.predict(z, V[2 * k:2 * k + 2], v_w)
        expected.append(z)
    np.testing.assert_allclose(qp.predict(z0, V, v_w), expected, atol=1e-12)


def test_quadratic_form_equals_rollout_cost():
    # J(V) − J(0) = ½ Vᵀ H V + fᵀ V checks H and f, including the terminal weight.
    rng = np.random.default_rng(1)
    A, B, D, C = random_model(rng)
    Q, R = np.diag([1.0, 2.0, 3.0, 4.0]), np.diag([0.5, 0.1])
    P = 10.0 * Q
    horizon = 4
    qp = condense(A, B, D, C, Q, R, horizon, P=P)
    z0, v_w = rng.standard_normal(6), rng.standard_normal(3)
    V = rng.standard_normal(horizon * 2)

    f = qp.linear_term(z0, v_w)
    J = rollout_cost(A, B, D, C, Q, R, P, z0, V, v_w, horizon)
    J0 = rollout_cost(A, B, D, C, Q, R, P, z0, np.zeros_like(V), v_w, horizon)
    assert 0.5 * V @ qp.H @ V + f @ V == pytest.approx(J - J0, rel=1e-10)


def test_quadratic_form_with_reference_equals_rollout_cost():
    # Same check as above with a per-step reference, so F_r is verified too.
    rng = np.random.default_rng(6)
    A, B, D, C = random_model(rng)
    Q, R = np.diag([1.0, 2.0, 3.0, 4.0]), np.diag([0.5, 0.1])
    P = 10.0 * Q
    horizon = 4
    qp = condense(A, B, D, C, Q, R, horizon, P=P)
    z0, v_w = rng.standard_normal(6), rng.standard_normal(3)
    r = rng.standard_normal((horizon, 4))
    V = rng.standard_normal(horizon * 2)

    f = qp.linear_term(z0, v_w, r)
    J = rollout_cost(A, B, D, C, Q, R, P, z0, V, v_w, horizon, r)
    J0 = rollout_cost(A, B, D, C, Q, R, P, z0, np.zeros_like(V), v_w, horizon, r)
    assert 0.5 * V @ qp.H @ V + f @ V == pytest.approx(J - J0, rel=1e-10)


def test_reference_at_rest_state_gives_zero_input():
    # At rest at p = 1, the target r = [1, 0] is already met: no input is optimal.
    # Without the reference the MPC would push towards p = 0.
    A, B, D, C = double_integrator()
    qp = condense(A, B, D, C, np.diag([10.0, 1.0]), np.array([[0.1]]), 5)
    z0, no_wind = np.array([1.0, 0.0]), np.zeros(1)
    with_ref = solve_qp(qp.H, qp.linear_term(z0, no_wind, r=[1.0, 0.0]))
    without_ref = solve_qp(qp.H, qp.linear_term(z0, no_wind))
    assert with_ref.solved and without_ref.solved
    np.testing.assert_allclose(with_ref.x, 0.0, atol=1e-12)
    assert without_ref.x[0] < -1e-3


def test_constant_reference_broadcasts_over_horizon():
    A, B, D, C = double_integrator()
    qp = condense(A, B, D, C, np.eye(2), np.eye(1), 3)
    z0, v_w, r = np.array([0.3, -0.2]), np.array([0.5]), np.array([1.0, 2.0])
    np.testing.assert_array_equal(qp.linear_term(z0, v_w, r),
                                  qp.linear_term(z0, v_w, np.tile(r, (3, 1))))


def test_linear_term_rejects_nan_reference():
    A, B, D, C = double_integrator()
    qp = condense(A, B, D, C, np.eye(2), np.eye(1), 3)
    with pytest.raises(ValueError, match="r contains NaN"):
        qp.linear_term(np.zeros(2), np.zeros(1), r=[np.nan, 0.0])


def test_dare_terminal_weight_reproduces_lqr():
    # With P from the DARE, the finite-horizon optimum is the infinite-horizon
    # LQR for any horizon, so ν_0 = −K z_0.
    A, B, D, C = double_integrator()
    Q, R = np.diag([10.0, 1.0]), np.array([[0.1]])
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    z0 = np.array([1.0, -0.5])
    for horizon in (1, 3, 10):
        qp = condense(A, B, D, C, Q, R, horizon, P=P)
        res = solve_qp(qp.H, qp.linear_term(z0, np.zeros(1)))
        assert res.solved
        np.testing.assert_allclose(res.x[0], -K @ z0, rtol=1e-6)


def test_unconstrained_solution_is_newton_step():
    rng = np.random.default_rng(2)
    A, B, D, C = random_model(rng)
    qp = condense(A, B, D, C, np.eye(4), np.eye(2), 5)
    f = qp.linear_term(rng.standard_normal(6), rng.standard_normal(3))
    res = solve_qp(qp.H, f)
    assert res.solved
    np.testing.assert_allclose(res.x, -np.linalg.solve(qp.H, f), rtol=1e-8, atol=1e-10)


def test_loose_bounds_give_the_unconstrained_solution():
    rng = np.random.default_rng(3)
    A, B, D, C = random_model(rng)
    qp = condense(A, B, D, C, np.eye(4), np.eye(2), 5)
    z0, v_w = rng.standard_normal(6), rng.standard_normal(3)
    f = qp.linear_term(z0, v_w)
    g_lb, g_ub = qp.state_bounds(z0, v_w, np.full(4, -1e6), np.full(4, 1e6))
    res = solve_qp(qp.H, f, lb=np.full(10, -1e6), ub=np.full(10, 1e6),
                   G=qp.G, g_lb=g_lb, g_ub=g_ub)
    assert res.solved
    np.testing.assert_allclose(res.x, -np.linalg.solve(qp.H, f), rtol=1e-8, atol=1e-10)


def test_tight_input_bound_is_active():
    A, B, D, C = double_integrator()
    qp = condense(A, B, D, C, np.diag([10.0, 1.0]), np.array([[0.1]]), 5)
    z0 = np.array([5.0, 0.0])  # far right: wants a large negative acceleration
    u_max = 0.2
    res = solve_qp(qp.H, qp.linear_term(z0, np.zeros(1)),
                   lb=np.full(5, -u_max), ub=np.full(5, u_max))
    assert res.solved
    assert res.x[0] == pytest.approx(-u_max, abs=1e-9)
    assert np.all(np.abs(res.x) <= u_max + 1e-9)


def test_state_constraint_is_respected_and_active():
    # Strong wind pushes the velocity up; the bound v ≤ 0.1 must hold on z_1..z_np.
    A, B, D, C = double_integrator()
    horizon = 5
    qp = condense(A, B, D, C, np.diag([1.0, 0.0]), np.array([[1e-3]]), horizon, E=C[1:])
    z0, v_w = np.zeros(2), np.array([3.0])
    v_max = 0.1
    unconstrained = solve_qp(qp.H, qp.linear_term(z0, v_w))
    assert qp.predict(z0, unconstrained.x, v_w)[:, 1].max() > v_max  # bound would bind

    g_lb, g_ub = qp.state_bounds(z0, v_w, [-np.inf], [v_max])
    res = solve_qp(qp.H, qp.linear_term(z0, v_w), G=qp.G, g_lb=g_lb, g_ub=g_ub)
    assert res.solved
    v = qp.predict(z0, res.x, v_w)[:, 1]
    assert v.max() <= v_max + 1e-8
    assert v.max() == pytest.approx(v_max, abs=1e-8)


def test_per_step_state_bounds_broadcast():
    A, B, D, C = double_integrator()
    qp = condense(A, B, D, C, np.eye(2), np.eye(1), 3)
    h_ub = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _, g_ub = qp.state_bounds(np.zeros(2), np.zeros(1), -h_ub, h_ub)
    np.testing.assert_array_equal(g_ub, h_ub.ravel())  # zero offset at z_0 = 0, no wind


def test_zero_error_and_zero_wind_give_zero_input():
    rng = np.random.default_rng(4)
    A, B, D, C = random_model(rng)
    qp = condense(A, B, D, C, np.eye(4), np.eye(2), 5)
    f = qp.linear_term(np.zeros(6), np.zeros(3))
    g_lb, g_ub = qp.state_bounds(np.zeros(6), np.zeros(3), -np.ones(4), np.ones(4))
    res = solve_qp(qp.H, f, lb=-np.ones(10), ub=np.ones(10), G=qp.G, g_lb=g_lb, g_ub=g_ub)
    assert res.solved
    np.testing.assert_allclose(res.x, 0.0, atol=1e-12)


def test_infeasible_problem_is_reported_not_solved():
    # Wind 3 m/s² for 0.5 s with |a| ≤ 0.1 cannot keep the velocity below 0.1.
    A, B, D, C = double_integrator()
    qp = condense(A, B, D, C, np.eye(2), np.eye(1), 5, E=C[1:])
    z0, v_w = np.zeros(2), np.array([3.0])
    g_lb, g_ub = qp.state_bounds(z0, v_w, [-np.inf], [0.1])
    res = solve_qp(qp.H, qp.linear_term(z0, v_w), lb=np.full(5, -0.1), ub=np.full(5, 0.1),
                   G=qp.G, g_lb=g_lb, g_ub=g_ub)
    assert not res.solved
    assert res.exitflag < 0


def test_warm_start_gives_the_same_solution():
    A, B, D, C = double_integrator()
    qp = condense(A, B, D, C, np.diag([10.0, 1.0]), np.array([[0.1]]), 5)
    f = qp.linear_term(np.array([5.0, 0.0]), np.zeros(1))
    cold = solve_qp(qp.H, f, lb=np.full(5, -0.2), ub=np.full(5, 0.2))
    warm = solve_qp(qp.H, f, lb=np.full(5, -0.2), ub=np.full(5, 0.2), x0=cold.x)
    assert warm.solved
    np.testing.assert_allclose(warm.x, cold.x, atol=1e-10)


@pytest.mark.parametrize("kwargs, match", [
    ({"R": np.zeros((2, 2))}, "positive definite"),
    ({"Q": -np.eye(4)}, "positive semidefinite"),
    ({"Q": np.triu(np.ones((4, 4)))}, "symmetric"),
    ({"horizon": 0}, "horizon"),
    ({"B": np.zeros((5, 2))}, "B must have shape"),
    ({"A": np.full((6, 6), np.nan)}, "NaN"),
])
def test_condense_rejects_bad_inputs(kwargs, match):
    rng = np.random.default_rng(5)
    A, B, D, C = random_model(rng)
    args = {"A": A, "B": B, "D": D, "C": C, "Q": np.eye(4), "R": np.eye(2), "horizon": 3}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        condense(**args)


def test_solve_qp_rejects_nan_bounds():
    with pytest.raises(ValueError, match="NaN"):
        solve_qp(np.eye(2), np.zeros(2), lb=np.array([0.0, np.nan]))


def random_ltv_model(rng, horizon, N=6, m=2, d=3):
    """Random time-varying matrices, each A_i with spectral radius 0.9."""
    As = rng.standard_normal((horizon, N, N))
    As *= 0.9 / np.abs(np.linalg.eigvals(As)).max(axis=1)[:, None, None]
    return As, rng.standard_normal((horizon, N, m)), rng.standard_normal((horizon, N, d))


def test_ltv_condense_with_constant_matrices_is_condense():
    rng = np.random.default_rng(3)
    A, B, D, C = random_model(rng)
    horizon, Q, R = 6, np.eye(4), 0.5 * np.eye(2)
    P = 2.0 * np.eye(4)
    lti = condense(A, B, D, C, Q, R, horizon, P=P, E=C[:2])
    ltv = condense_ltv(np.repeat(A[None], horizon, 0), np.repeat(B[None], horizon, 0),
                       np.repeat(D[None], horizon, 0), C, Q, R, P=P, E=C[:2])
    for name in ("Phi", "Gamma", "Lambda", "H", "F_z", "F_w", "F_r", "G", "E_bar"):
        np.testing.assert_allclose(getattr(ltv, name), getattr(lti, name), atol=1e-10,
                                   err_msg=name)
    assert ltv.horizon == horizon


def test_ltv_prediction_and_cost_match_a_time_varying_rollout():
    rng = np.random.default_rng(4)
    horizon = 5
    As, Bs, Ds = random_ltv_model(rng, horizon)
    C = np.hstack([np.eye(4), np.zeros((4, 2))])
    Q, R, P = np.diag([1.0, 2.0, 3.0, 4.0]), np.diag([0.3, 0.7]), 5.0 * np.eye(4)
    qp = condense_ltv(As, Bs, Ds, C, Q, R, P=P)
    z0, V, v = rng.standard_normal(6), rng.standard_normal(horizon * 2), rng.standard_normal(3)

    z, J, states = z0, 0.0, []
    for i in range(horizon):
        nu = V[2 * i:2 * i + 2]
        z = As[i] @ z + Bs[i] @ nu + Ds[i] @ v
        states.append(z)
        W = P if i == horizon - 1 else Q
        J += nu @ R @ nu + (C @ z) @ W @ (C @ z)
    np.testing.assert_allclose(qp.predict(z0, V, v), np.array(states), atol=1e-10)
    # ½ Vᵀ H V + fᵀ V equals J up to the V-independent term.
    J0 = sum((C @ s) @ (P if i == horizon - 1 else Q) @ (C @ s)
             for i, s in enumerate(qp.predict(z0, np.zeros_like(V), v)))
    quadratic = 0.5 * V @ qp.H @ V + qp.linear_term(z0, v) @ V
    assert quadratic + J0 == pytest.approx(J, rel=1e-10)


def test_ltv_affine_term_enters_as_a_unit_disturbance():
    rng = np.random.default_rng(5)
    horizon = 4
    As, Bs, _ = random_ltv_model(rng, horizon)
    offsets = rng.standard_normal((horizon, 6))
    C = np.eye(6)
    qp = condense_ltv(As, Bs, offsets[:, :, None], C, np.eye(6), np.eye(2))
    z = z0 = rng.standard_normal(6)
    expected = []
    for i in range(horizon):
        z = As[i] @ z + offsets[i]
        expected.append(z)
    np.testing.assert_allclose(qp.predict(z0, np.zeros(2 * horizon), np.ones(1)),
                               np.array(expected), atol=1e-10)


def test_ltv_condense_rejects_inconsistent_shapes():
    rng = np.random.default_rng(6)
    As, Bs, Ds = random_ltv_model(rng, 3)
    C = np.eye(6)
    with pytest.raises(ValueError, match="Bs"):
        condense_ltv(As, Bs[:2], Ds, C, np.eye(6), np.eye(2))
    with pytest.raises(ValueError, match="As"):
        condense_ltv(As[0], Bs, Ds, C, np.eye(6), np.eye(2))
    bad = As.copy()
    bad[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        condense_ltv(bad, Bs, Ds, C, np.eye(6), np.eye(2))
