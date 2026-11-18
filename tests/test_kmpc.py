"""K-MPC: physical-units problem solved in standardized lifted coordinates.

The test model is exact: a physical double integrator e⁺ = A_e e + B_e u + D_e w,
rewritten for z = (e − μ)/σ and the affine lifting ψ = [z, 1]. So K-MPC must
reproduce the QP posed directly on the physical system, which checks every
coordinate change (SQS, r = −μ/σ, scaled bounds, un-scaled predictions).
"""
import numpy as np
import pytest

from controllers.kmpc import KMPCSettings, KoopmanMPC
from controllers.qp import condense, solve_qp
from koopman.lifting import MonomialLifting
from koopman.models import KoopmanModel
from koopman.preprocessing import Standardizer

DT = 0.1
# e = [p (m), v (m/s)], u = acceleration (m/s²), wind enters as an acceleration.
A_E = np.array([[1.0, DT], [0.0, 1.0]])
B_E = np.array([[0.5 * DT**2], [DT]])
D_E = B_E.copy()
# Non-zero mean on purpose: z = 0 is e = μ, not zero error.
SCALER = Standardizer(mean=np.array([0.3, -0.1]), std=np.array([0.5, 2.0]))
HORIZON = 5


def standardized_model(scaler=SCALER) -> KoopmanModel:
    """Exact model of the physical system in ψ = [z, 1] coordinates."""
    S_inv, S = np.diag(1 / scaler.std), np.diag(scaler.std)
    A = np.zeros((3, 3))
    A[:2, :2] = S_inv @ A_E @ S
    A[:2, 2] = S_inv @ (A_E - np.eye(2)) @ scaler.mean
    A[2, 2] = 1.0
    B = np.vstack([S_inv @ B_E, np.zeros((1, 1))])
    D = np.vstack([S_inv @ D_E, np.zeros((1, 1))])
    return KoopmanModel(A=A, B=B, D=D, C=np.eye(2, 3))


def settings(**overrides) -> KMPCSettings:
    values = dict(Q=np.diag([10.0, 1.0]), R=np.array([[0.1]]), horizon=HORIZON,
                  u_lb=[-np.inf], u_ub=[np.inf], h_lb=[-np.inf, -np.inf],
                  h_ub=[np.inf, np.inf], wind_mean=[0.0])
    values.update(overrides)
    return KMPCSettings(**values)


def make_mpc(**overrides) -> KoopmanMPC:
    return KoopmanMPC(standardized_model(), MonomialLifting(2, 1), SCALER, settings(**overrides))


def physical_rollout(e0, inputs, wind):
    e, out = np.asarray(e0, dtype=float), []
    for u in inputs:
        e = A_E @ e + B_E @ u + D_E @ wind
        out.append(e)
    return np.array(out)


def test_zero_error_and_zero_wind_give_zero_input_despite_nonzero_mean():
    # Regression for the standardization offset: penalizing z instead would drive e to μ, not 0.
    step = make_mpc().solve(np.zeros(2))
    assert step.solved
    np.testing.assert_allclose(step.u_d, 0.0, atol=1e-12)
    np.testing.assert_allclose(step.predicted_error, 0.0, atol=1e-12)


@pytest.mark.parametrize("wind", [0.0, 1.5])
def test_matches_qp_posed_on_the_physical_system(wind):
    s = settings(wind_mean=[wind])
    e0 = np.array([0.8, -0.4])
    step = make_mpc(wind_mean=[wind]).solve(e0)

    physical = condense(A_E, B_E, D_E, np.eye(2), s.Q, s.R, HORIZON)
    direct = solve_qp(physical.H, physical.linear_term(e0, np.array([wind])))
    assert step.solved and direct.solved
    np.testing.assert_allclose(step.planned_inputs.ravel(), direct.x, rtol=1e-8, atol=1e-10)


def test_predicted_error_matches_physical_rollout_of_the_plan():
    wind = np.array([0.7])
    e0 = np.array([0.5, 0.2])
    step = make_mpc(wind_mean=wind).solve(e0)
    np.testing.assert_allclose(step.predicted_error,
                               physical_rollout(e0, step.planned_inputs, wind), atol=1e-10)


def test_input_bounds_are_respected():
    u_max = 0.2
    step = make_mpc(u_lb=[-u_max], u_ub=[u_max]).solve(np.array([5.0, 0.0]))
    assert step.solved
    assert step.u_d[0] == pytest.approx(-u_max, abs=1e-9)
    assert np.all(np.abs(step.planned_inputs) <= u_max + 1e-9)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_physical_state_bound_is_respected(sign):
    # Wind pushes v towards a physical bound |v| ≤ 0.3 m/s, scaled inside. With
    # μ_v = −0.1, σ_v = 2 the scaled bounds are 0.2 and −0.1, so an unscaled
    # bound would land elsewhere.
    wind, v_max = np.array([3.0 * sign]), 0.3
    weights = dict(Q=np.diag([1.0, 0.0]), R=np.array([[1e-3]]), wind_mean=wind)
    free = make_mpc(**weights).solve(np.zeros(2)).predicted_error[:, 1]
    assert np.abs(free).max() > v_max  # the bound would bind

    mpc = make_mpc(**weights, h_lb=[-np.inf, -v_max], h_ub=[np.inf, v_max])
    step = mpc.solve(np.zeros(2))
    assert step.solved
    assert (sign * step.predicted_error[:, 1]).max() == pytest.approx(v_max, abs=1e-8)


def test_infeasible_problem_falls_back_to_zero_input():
    # |u| ≤ 0.1 cannot hold v ≤ 0.1 against a 3 m/s² wind.
    wind = np.array([3.0])
    mpc = make_mpc(u_lb=[-0.1], u_ub=[0.1], h_ub=[np.inf, 0.1], wind_mean=wind)
    step = mpc.solve(np.zeros(2))
    assert not step.solved
    assert step.exitflag < 0
    np.testing.assert_array_equal(step.u_d, 0.0)
    np.testing.assert_array_equal(step.planned_inputs, 0.0)
    # The prediction is for what is applied: the LQR alone, i.e. ν = 0.
    np.testing.assert_allclose(step.predicted_error,
                               physical_rollout(np.zeros(2), np.zeros((HORIZON, 1)), wind),
                               atol=1e-10)


def test_warm_start_does_not_change_the_solution():
    mpc = make_mpc(u_lb=[-0.2], u_ub=[0.2])
    e0, e1 = np.array([2.0, 0.0]), np.array([1.9, -0.2])
    mpc.solve(e0)
    warm = mpc.solve(e1)
    cold = make_mpc(u_lb=[-0.2], u_ub=[0.2]).solve(e1)
    np.testing.assert_allclose(warm.planned_inputs, cold.planned_inputs, atol=1e-10)
    mpc.reset()
    np.testing.assert_allclose(mpc.solve(e1).planned_inputs, cold.planned_inputs, atol=1e-10)


def test_solve_rejects_bad_error_state():
    mpc = make_mpc()
    with pytest.raises(ValueError, match="shape"):
        mpc.solve(np.zeros(3))
    with pytest.raises(ValueError, match="NaN"):
        mpc.solve(np.array([np.nan, 0.0]))


def test_rejects_model_2():
    model = standardized_model()
    model2 = KoopmanModel(A=model.A, B=model.B, D=model.D, C=model.C, use_residual=True)
    with pytest.raises(ValueError, match="Model 1"):
        KoopmanMPC(model2, MonomialLifting(2, 1), SCALER, settings())


@pytest.mark.parametrize("lifting, scaler, overrides, match", [
    (MonomialLifting(2, 2), SCALER, {}, "lifting maps"),
    (MonomialLifting(2, 1), Standardizer(np.zeros(3), np.ones(3)), {}, "channels"),
    (MonomialLifting(2, 1), SCALER, {"wind_mean": [0.0, 0.0]}, "wind_mean"),
    (MonomialLifting(2, 1), SCALER, {"h_lb": [0.0], "h_ub": [1.0]}, "h_lb"),
])
def test_rejects_mismatched_dimensions(lifting, scaler, overrides, match):
    with pytest.raises(ValueError, match=match):
        KoopmanMPC(standardized_model(), lifting, scaler, settings(**overrides))


@pytest.mark.parametrize("overrides, match", [
    ({"u_lb": [1.0], "u_ub": [0.0]}, "must not exceed"),
    ({"h_lb": [np.nan, 0.0]}, "NaN"),
    ({"wind_mean": [np.inf]}, "NaN or inf"),
])
def test_settings_reject_bad_values(overrides, match):
    with pytest.raises(ValueError, match=match):
        settings(**overrides)
