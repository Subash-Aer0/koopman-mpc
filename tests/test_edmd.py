"""Tests for edmd and KoopmanModel: snapshot layout, exact recovery, Lasso optimality."""
import numpy as np
import pytest

from koopman.edmd import build_snapshots, fit_edmd
from koopman.lifting import MonomialLifting
from koopman.models import KoopmanModel


def simulate_linear(A, B, D, x0, u, w):
    """x_k+1 = A x_k + B u_k + D w_k, returning (T, n) with T = len(u)."""
    x = [x0]
    for k in range(len(u) - 1):
        x.append(A @ x[-1] + B @ u[k] + D @ w[k])
    return np.array(x)


def test_snapshots_never_cross_episodes():
    lifting = MonomialLifting(1, 1)  # ψ = [x, 1]
    errors = [np.array([[0.0], [1.0], [2.0]]), np.array([[10.0], [11.0]])]
    inputs = [np.full((3, 1), 5.0), np.full((2, 1), 6.0)]
    winds = [np.full((3, 1), 7.0), np.full((2, 1), 8.0)]
    gamma1, gamma2 = build_snapshots(errors, inputs, winds, lifting)

    assert gamma1.shape == gamma2.shape == (4, 3)  # (2 + 1 + 1) rows, 2 + 1 pairs
    np.testing.assert_array_equal(gamma1[0], [0.0, 1.0, 10.0])
    np.testing.assert_array_equal(gamma2[0], [1.0, 2.0, 11.0])  # never (2.0 -> 10.0)
    np.testing.assert_array_equal(gamma1[2:], [[5.0, 5.0, 6.0], [7.0, 7.0, 8.0]])
    np.testing.assert_array_equal(gamma2[2:], 0.0)


def test_snapshots_reject_single_sample_episode():
    lifting = MonomialLifting(1, 1)
    with pytest.raises(ValueError, match="at least 2"):
        build_snapshots([np.zeros((1, 1))], [np.zeros((1, 1))], [np.zeros((1, 1))], lifting)


def test_least_squares_recovers_linear_system_exactly():
    # ψ = [x, 1] and data from an exactly linear system: [A B D] is in the model class.
    rng = np.random.default_rng(0)
    n, m, d = 3, 2, 2
    A_x = 0.9 * np.linalg.qr(rng.standard_normal((n, n)))[0]
    B_x, D_x = rng.standard_normal((n, m)), rng.standard_normal((n, d))
    errors, inputs, winds = [], [], []
    for _ in range(3):
        u, w = rng.standard_normal((40, m)), rng.standard_normal((40, d))
        errors.append(simulate_linear(A_x, B_x, D_x, rng.standard_normal(n), u, w))
        inputs.append(u)
        winds.append(w)

    lifting = MonomialLifting(n, 1)
    gamma1, gamma2 = build_snapshots(errors, inputs, winds, lifting)
    A, B, D = fit_edmd(gamma1, gamma2, lifting.lifted_dim, m, d, 0.0)

    np.testing.assert_allclose(A[:n, :n], A_x, atol=1e-10)
    np.testing.assert_allclose(B[:n], B_x, atol=1e-10)
    np.testing.assert_allclose(D[:n], D_x, atol=1e-10)
    # The constant monomial maps to itself: ψ_const,k+1 = 1.
    np.testing.assert_allclose(A[n], np.eye(n + 1)[n], atol=1e-10)


def test_lasso_matches_soft_threshold_for_orthogonal_regressor():
    # With Γ1 Γ1ᵀ = I, ||Y − MΓ1||² + λ||M||₁ has the closed form
    # M = soft(Y Γ1ᵀ, λ/2), elementwise.
    rng = np.random.default_rng(1)
    gamma1 = np.linalg.qr(rng.standard_normal((50, 4)))[0].T  # (4, 50), orthonormal rows
    Y = rng.standard_normal((2, 50))
    gamma2 = np.vstack([Y, np.zeros((2, 50))])
    l1_lambda = 0.8
    A, B, D = fit_edmd(gamma1, gamma2, 2, 1, 1, l1_lambda)

    unregularized = Y @ gamma1.T
    expected = np.sign(unregularized) * np.maximum(np.abs(unregularized) - l1_lambda / 2, 0.0)
    np.testing.assert_allclose(np.hstack([A, B, D]), expected, atol=1e-8)


def test_lasso_shrinks_towards_zero():
    rng = np.random.default_rng(2)
    gamma1 = rng.standard_normal((5, 200))
    gamma2 = np.vstack([rng.standard_normal((3, 200)), np.zeros((2, 200))])
    norms = [np.abs(np.hstack(fit_edmd(gamma1, gamma2, 3, 1, 1, lam))).sum()
             for lam in (0.0, 10.0, 100.0)]
    assert norms[0] > norms[1] > norms[2]


def lifted_lasso_data(l1_lambda=5.0):
    """Γ1, Γ2 from a linear system lifted with degree-2 monomials, n = 2 (N = 6)."""
    rng = np.random.default_rng(4)
    n, m, d = 2, 1, 1
    A_x = np.array([[0.9, 0.1], [-0.1, 0.8]])
    B_x, D_x = rng.standard_normal((n, m)), rng.standard_normal((n, d))
    errors, inputs, winds = [], [], []
    for _ in range(3):
        u, w = rng.standard_normal((60, m)), rng.standard_normal((60, d))
        errors.append(simulate_linear(A_x, B_x, D_x, rng.standard_normal(n), u, w))
        inputs.append(u)
        winds.append(w)
    lifting = MonomialLifting(n, 2)
    gamma1, gamma2 = build_snapshots(errors, inputs, winds, lifting)
    return gamma1, gamma2, lifting, m, d


def test_lasso_shrinks_the_constant_row_when_not_fixed():
    # The failure mode the fix exists for: the "1" no longer maps to itself.
    gamma1, gamma2, lifting, m, d = lifted_lasso_data()
    A, _, _ = fit_edmd(gamma1, gamma2, lifting.lifted_dim, m, d, 5.0)
    assert A[lifting.constant_index, lifting.constant_index] < 1.0 - 1e-3


@pytest.mark.parametrize("l1_lambda", [0.0, 5.0])
def test_fixed_constant_row_is_exact_identity(l1_lambda):
    gamma1, gamma2, lifting, m, d = lifted_lasso_data()
    j = lifting.constant_index
    A, B, D = fit_edmd(gamma1, gamma2, lifting.lifted_dim, m, d, l1_lambda, constant_index=j)
    np.testing.assert_array_equal(A[j], np.eye(lifting.lifted_dim)[j])
    np.testing.assert_array_equal(B[j], 0.0)
    np.testing.assert_array_equal(D[j], 0.0)


def test_fixing_the_constant_row_leaves_other_rows_unchanged():
    # The objective separates by rows, so the other rows are the unconstrained fit.
    gamma1, gamma2, lifting, m, d = lifted_lasso_data()
    j, N = lifting.constant_index, lifting.lifted_dim
    free = np.hstack(fit_edmd(gamma1, gamma2, N, m, d, 5.0))
    fixed = np.hstack(fit_edmd(gamma1, gamma2, N, m, d, 5.0, constant_index=j))
    np.testing.assert_allclose(np.delete(fixed, j, axis=0), np.delete(free, j, axis=0),
                               atol=1e-10)


@pytest.mark.parametrize("index, match", [(0, "not 1 in every snapshot"),
                                          (6, r"\[0, 6\)"), (-1, r"\[0, 6\)")])
def test_fit_rejects_wrong_constant_index(index, match):
    gamma1, gamma2, lifting, m, d = lifted_lasso_data()
    with pytest.raises(ValueError, match=match):
        fit_edmd(gamma1, gamma2, lifting.lifted_dim, m, d, 5.0, constant_index=index)


def test_fit_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match=r"\(6, K\)"):
        fit_edmd(np.zeros((5, 10)), np.zeros((5, 10)), 3, 2, 1, 0.0)


def test_model_predict_and_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    model = KoopmanModel(A=rng.standard_normal((4, 4)), B=rng.standard_normal((4, 2)),
                         D=rng.standard_normal((4, 3)), C=np.eye(2, 4))
    psi, u, w = rng.standard_normal(4), rng.standard_normal(2), rng.standard_normal(3)
    np.testing.assert_allclose(model.predict(psi, u, w), model.A @ psi + model.B @ u + model.D @ w)

    model.save(tmp_path / "model.npz")
    loaded = KoopmanModel.load(tmp_path / "model.npz")
    for name in ("A", "B", "D", "C"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(model, name))
    assert loaded.use_residual is False


def test_model1_rejects_residual_input():
    model = KoopmanModel(A=np.eye(2), B=np.zeros((2, 1)), D=np.zeros((2, 1)), C=np.eye(1, 2))
    with pytest.raises(ValueError, match="Model 1"):
        model.predict(np.zeros(2), np.zeros(1), np.zeros(1), u_res=np.zeros(1))
