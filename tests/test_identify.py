"""Identification plumbing: standardization, Γ1 diagnostics, rollout and the 02 script.

The script is exercised with minimal stand-ins for ψ, EDMD and KoopmanModel,
so these tests check the wiring around the Koopman modules, not their math.
"""
from dataclasses import dataclass
import importlib.util
import json

import numpy as np
import pytest

from conftest import REPO_ROOT
from datagen.collect import Split
from experiment.builders import load_identify
from koopman.preprocessing import Standardizer, regressor_diagnostics
from test_datagen import load_script as load_datagen_script
from test_datagen import write_datagen

IDENTIFY = REPO_ROOT / "configs" / "identify" / "model1.toml"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "identify_models", REPO_ROOT / "scripts" / "02_identify_models.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


class AffineLifting:
    """ψ(z) = [z, 1]: degree-1 stand-in honouring the state-first contract."""

    def __init__(self, state_dim, degree):
        self.state_dim = state_dim

    @property
    def lifted_dim(self):
        return self.state_dim + 1

    @property
    def constant_index(self):
        return self.state_dim

    def __call__(self, x):
        return np.hstack([x, np.ones((len(x), 1))])


@dataclass
class LinearModel:
    A: np.ndarray
    B: np.ndarray
    D: np.ndarray
    C: np.ndarray
    use_residual: bool = False

    def predict(self, psi, u_d, v_w, u_res=None):
        return self.A @ psi + self.B @ u_d + self.D @ v_w

    def save(self, path):
        np.savez(path, A=self.A, B=self.B, D=self.D, C=self.C)


def stand_in_snapshots(errors, inputs, winds, lifting):
    gamma1 = np.hstack([np.vstack([lifting(z[:-1]).T, u[:-1].T, w[:-1].T])
                        for z, u, w in zip(errors, inputs, winds)])
    gamma2 = np.hstack([np.vstack([lifting(z[1:]).T, np.zeros((u.shape[1] + w.shape[1],
                                                                len(z) - 1))])
                        for z, u, w in zip(errors, inputs, winds)])
    return gamma1, gamma2


def stand_in_fit(gamma1, gamma2, state_dim, input_dim, dist_dim, l1_lambda, constant_index=None):
    M = np.linalg.lstsq(gamma1.T, gamma2[:state_dim].T, rcond=None)[0].T
    return M[:, :state_dim], M[:, state_dim:state_dim + input_dim], M[:, state_dim + input_dim:]


# --- standardization and diagnostics -----------------------------------------

def test_standardizer_round_trips_and_whitens():
    data = np.random.default_rng(0).normal([1.0, -2.0, 0.0], [0.1, 3.0, 0.01], (4, 500, 3))
    scaler = Standardizer.fit(data)
    z = scaler.transform(data)
    np.testing.assert_allclose(z.reshape(-1, 3).mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(z.reshape(-1, 3).std(axis=0), 1.0)
    np.testing.assert_allclose(scaler.inverse(z), data)


def test_standardizer_rejects_constant_channel():
    data = np.random.default_rng(0).standard_normal((100, 3))
    data[:, 1] = 2.0
    with pytest.raises(ValueError, match=r"channels \[1\]"):
        Standardizer.fit(data)


def test_diagnostics_full_rank():
    gamma1 = np.random.default_rng(0).standard_normal((5, 50))
    result = regressor_diagnostics(gamma1, 5)
    assert result["rank"] == 5 and result["columns"] == 50
    assert result["condition_number"] >= 1.0


def test_diagnostics_reject_collinear_rows():
    # A constant wind channel duplicates the constant monomial: D is unidentifiable.
    gamma1 = np.random.default_rng(0).standard_normal((5, 50))
    gamma1[4] = 3.0 * gamma1[3]
    with pytest.raises(ValueError, match="rank 4 < 5"):
        regressor_diagnostics(gamma1, 5)


def test_diagnostics_reject_too_few_snapshots():
    with pytest.raises(ValueError, match="3 snapshots"):
        regressor_diagnostics(np.ones((5, 3)), 5)


def test_shipped_identify_config_matches_paper():
    cfg = load_identify(IDENTIFY)
    assert (cfg.degree, cfg.l1_lambda, cfg.horizon) == (2, 20.0, 5)


# --- script helpers ----------------------------------------------------------

def test_projection_rejects_lifting_without_state_first():
    script = load_script()

    class StateLast(AffineLifting):
        def __call__(self, x):
            return np.hstack([np.ones((len(x), 1)), x])

    np.testing.assert_array_equal(script.projection(AffineLifting(3, 1), 3)[:, :3], np.eye(3))
    with pytest.raises(ValueError, match="state first"):
        script.projection(StateLast(3, 1), 3)


def test_rollout_matches_known_linear_system():
    # ψ = [z, 1], z_k+1 = 0.5 z_k + u_k: analytic z_k = 0.5^k z_0 + Σ 0.5^(k−1−j) u_j.
    script = load_script()
    lifting = AffineLifting(2, 1)
    A = np.diag([0.5, 0.5, 1.0])
    B = np.vstack([np.eye(2), np.zeros((1, 2))])
    model = LinearModel(A=A, B=B, D=np.zeros((3, 1)), C=None)
    C = script.projection(lifting, 2)
    u = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
    z = script.rollout(model, lifting, C, np.array([4.0, -4.0]), u, np.zeros((4, 1)))
    np.testing.assert_allclose(z, [[4.0, -4.0], [3.0, -2.0], [1.5, 1.0], [0.75, 0.5]])


def error_only_split(e_x, u_d, v_w) -> Split:
    """Split with the given e_x, u_d, v_w (E, T, ·); raw-state fields are zeros."""
    E, T = e_x.shape[:2]
    return Split(t=0.1 * np.arange(T), reference=np.array(["r"] * E), repeat=np.arange(E),
                 e_x=e_x, u_d=u_d, v_w=v_w, x=np.zeros((E, T, 13)), x_ref=np.zeros((E, T, 10)),
                 u_h=np.zeros((E, T, 4)), u_cmd=np.zeros((E, T, 4)))


def linear_error_data(rng, A_e, B_e, D_e, episodes=2, T=30):
    """e_x from e_k+1 = A_e e_k + B_e u_k + D_e w_k, as (E, T, ·) arrays."""
    e_x = np.zeros((episodes, T, 9))
    u_d, v_w = rng.standard_normal((episodes, T, 4)), rng.standard_normal((episodes, T, 3))
    e_x[:, 0] = rng.standard_normal((episodes, 9))
    for k in range(T - 1):
        e_x[:, k + 1] = e_x[:, k] @ A_e.T + u_d[:, k] @ B_e.T + v_w[:, k] @ D_e.T
    return e_x, u_d, v_w


def lifted_model(A_e, B_e, D_e, C):
    """The exact system in ψ = [e, 1] coordinates; the constant maps to itself."""
    A = np.zeros((10, 10))
    A[:9, :9], A[9, 9] = A_e, 1.0
    return LinearModel(A=A, B=np.vstack([B_e, np.zeros((1, 4))]),
                       D=np.vstack([D_e, np.zeros((1, 3))]), C=C)


@pytest.mark.parametrize("horizon", [1, 5])
def test_horizon_errors_vanish_for_exact_model(horizon):
    script = load_script()
    rng = np.random.default_rng(0)
    A_e = 0.9 * np.linalg.qr(rng.standard_normal((9, 9)))[0]
    B_e, D_e = rng.standard_normal((9, 4)), rng.standard_normal((9, 3))
    split = error_only_split(*linear_error_data(rng, A_e, B_e, D_e))
    lifting = AffineLifting(9, 1)
    model = lifted_model(A_e, B_e, D_e, script.projection(lifting, 9))
    identity = Standardizer(mean=np.zeros(9), std=np.ones(9))

    errors = script.horizon_errors(model, lifting, model.C, identity, split, horizon)
    assert max(err["rmse"] for err in errors.values()) < 1e-10


def test_horizon_errors_match_loop_for_wrong_model():
    # Vectorized starts must equal a plain loop: H-step from each k with model.predict.
    script = load_script()
    rng = np.random.default_rng(1)
    A_e = 0.9 * np.eye(9)
    split = error_only_split(*linear_error_data(rng, A_e, np.ones((9, 4)), np.ones((9, 3))))
    lifting = AffineLifting(9, 1)
    model = lifted_model(0.5 * A_e, np.zeros((9, 4)), np.ones((9, 3)), script.projection(lifting, 9))
    scaler = Standardizer.fit(split.e_x)
    H = 3

    predicted, true = [], []
    for e_i, u_i, w_i in zip(split.e_x, split.u_d, split.v_w):
        for k in range(len(e_i) - H):
            psi = lifting(scaler.transform(e_i[k])[None, :])[0]
            for j in range(H):
                psi = model.predict(psi, u_i[k + j], w_i[k + j])
            predicted.append(scaler.inverse(model.C @ psi))
            true.append(e_i[k + H])
    expected = script.rollout_errors(np.array(predicted)[None], np.array(true)[None])
    errors = script.horizon_errors(model, lifting, model.C, scaler, split, H)
    for label in script.ERROR_LABELS:
        assert errors[label]["rmse"] == pytest.approx(expected[label]["rmse"], rel=1e-10)


def test_horizon_predictions_align_with_truth_and_pad_with_nan():
    # Exact model: the prediction stored at k (made from k − H) equals e_x[k].
    script = load_script()
    rng = np.random.default_rng(2)
    A_e = 0.9 * np.linalg.qr(rng.standard_normal((9, 9)))[0]
    B_e, D_e = rng.standard_normal((9, 4)), rng.standard_normal((9, 3))
    split = error_only_split(*linear_error_data(rng, A_e, B_e, D_e))
    lifting = AffineLifting(9, 1)
    model = lifted_model(A_e, B_e, D_e, script.projection(lifting, 9))
    identity = Standardizer(mean=np.zeros(9), std=np.ones(9))

    predicted = script.horizon_predictions(model, lifting, model.C, identity, split, 4)
    assert predicted.shape == split.e_x.shape
    assert np.all(np.isnan(predicted[:, :4]))
    np.testing.assert_allclose(predicted[:, 4:], split.e_x[:, 4:], atol=1e-10)


def test_horizon_errors_reject_out_of_range_horizon():
    script = load_script()
    split = error_only_split(np.ones((1, 5, 9)), np.zeros((1, 5, 4)), np.zeros((1, 5, 3)))
    model = lifted_model(np.eye(9), np.zeros((9, 4)), np.zeros((9, 3)), np.eye(9, 10))
    for horizon in (0, 5):
        with pytest.raises(ValueError, match="horizon"):
            script.horizon_errors(model, AffineLifting(9, 1), model.C,
                                  Standardizer(np.zeros(9), np.ones(9)), split, horizon)


# --- script end to end -------------------------------------------------------

def test_identify_script_runs_with_stand_ins(tmp_path, monkeypatch):
    dataset_dir = load_datagen_script().main(
        [str(write_datagen(tmp_path, duration=2.0)), "--results-dir", str(tmp_path / "data")])

    script = load_script()
    monkeypatch.setattr(script, "MonomialLifting", AffineLifting)
    monkeypatch.setattr(script, "KoopmanModel", LinearModel)
    monkeypatch.setattr(script, "build_snapshots", stand_in_snapshots)
    monkeypatch.setattr(script, "fit_edmd", stand_in_fit)
    run_dir = script.main([str(dataset_dir), str(IDENTIFY), "--results-dir", str(tmp_path / "id")])

    report = json.loads((run_dir / "identification.json").read_text())
    assert report["lifted_dim"] == 10
    # 2 training episodes x 19 pairs; 10 lifted + 4 inputs + 3 wind regressors.
    assert report["regressor"]["columns"] == 38
    assert report["regressor"]["rank"] == 17
    assert set(report["validation_rollout"]) == set(script.ERROR_LABELS)
    with np.load(run_dir / "model.npz") as model:
        assert model["A"].shape == (10, 10)
    with np.load(run_dir / "standardizer.npz") as scaler:
        assert scaler["mean"].shape == (9,)


def test_identify_script_runs_with_real_koopman_modules(tmp_path):
    # Degree 1 so the short dataset (38 pairs) still exceeds the 17 regressors.
    dataset_dir = load_datagen_script().main(
        [str(write_datagen(tmp_path, duration=2.0)), "--results-dir", str(tmp_path / "data")])
    config = tmp_path / "identify.toml"
    config.write_text('[identify]\nname = "tiny_fit"\nseed = 0\ndegree = 1\nl1_lambda = 0.5\nhorizon = 3\n')

    run_dir = load_script().main([str(dataset_dir), str(config),
                                  "--results-dir", str(tmp_path / "id")])
    report = json.loads((run_dir / "identification.json").read_text())
    assert report["regressor"]["rank"] == 17
    assert report["validation_horizon_steps"] == 3
    assert set(report["validation_horizon"]) == set(load_script().ERROR_LABELS)
    for mode in ("1step", "3step", "rollout"):
        assert (run_dir / "figures" / f"prediction_circle_0_{mode}.png").exists()
    assert set(report["validation_one_step"]) == set(load_script().ERROR_LABELS)
    with np.load(run_dir / "validation_prediction.npz") as data:
        assert data["predicted"].shape == data["true"].shape == (1, 20, 9)
        # The rollout starts from the logged first sample.
        np.testing.assert_allclose(data["predicted"][:, 0], data["true"][:, 0], atol=1e-12)
