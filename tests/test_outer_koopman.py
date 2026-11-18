"""Selective lifting, PE check, output MPC and the outer-loop Koopman model."""
import numpy as np
import pytest

from controllers.hover_model import HoverModel
from controllers.linear_mpc import LinearizedReference
from controllers.output_mpc import OutputMPC, OutputMPCSettings
from dynamics import dynamics as rigid_body
from dynamics.params import QuadParams
from dynamics.quaternion import Quaternion
from koopman.edmd import build_snapshots, fit_edmd
from koopman.excitation import hankel_matrix, persistency_of_excitation
from koopman.outer_model import OuterKoopmanModel, deviation_state, horizon_rmse
from koopman.selective_lifting import SelectiveMonomialLifting
from koopman.trig_lifting import TrigMonomialLifting as TrigLifting

RATE = 2.0 / np.sqrt(3.0)


@pytest.fixture
def vehicle() -> QuadParams:
    return QuadParams(mass=1.0, inertia=np.diag([0.0082, 0.0082, 0.0149]),
                      drag_coeffs=np.array([0.05, 0.05, 0.10]), gravity=9.81)


def settings() -> OutputMPCSettings:
    return OutputMPCSettings(Q=np.diag([100.0] * 3 + [10.0] * 3 + [1.0] * 3),
                             R=np.diag([1.0, 5.0, 5.0, 0.1]), horizon=20,
                             u_lb=np.array([-RATE] * 3 + [2.0]), u_ub=np.array([RATE] * 3 + [18.0]))


def full_state(x9, omega=np.zeros(3)) -> np.ndarray:
    x = np.zeros(rigid_body.STATE_DIM)
    x[rigid_body.POS] = x9[:3]
    x[rigid_body.VEL] = x9[3:6]
    x[rigid_body.QUAT] = Quaternion.from_euler(x9[6:9]).to_numpy()
    x[rigid_body.RATE] = omega
    return x


# --- lifting -----------------------------------------------------------------

def test_selective_lifting_layout_and_size():
    lifting = SelectiveMonomialLifting(4, 2, nonlinear_indices=[2, 3])
    x = np.array([[1.0, 2.0, 3.0, 5.0]])
    psi = lifting(x)[0]
    np.testing.assert_allclose(psi[:4], x[0])        # state first: C = [I 0]
    assert psi[lifting.constant_index] == 1.0
    np.testing.assert_allclose(psi[5:], [9.0, 15.0, 25.0])   # x3², x3 x4, x4²
    assert lifting.lifted_dim == 4 + 1 + 3
    assert lifting.feature_names(["a", "b", "c", "d"])[5:] == ["c*c", "c*d", "d*d"]


def test_quadrotor_dictionary_has_no_position_monomials():
    lifting = SelectiveMonomialLifting(12, 2, nonlinear_indices=range(3, 9))
    assert lifting.lifted_dim == 12 + 1 + 21
    assert not lifting.exponents[13:, :3].any()


def test_selective_lifting_rejects_bad_indices():
    with pytest.raises(ValueError):
        SelectiveMonomialLifting(3, 2, nonlinear_indices=[0, 0])
    with pytest.raises(ValueError):
        SelectiveMonomialLifting(3, 2, nonlinear_indices=[3])


# --- persistency of excitation -----------------------------------------------

def test_hankel_matrix_layout():
    u = np.arange(10.0).reshape(5, 2)
    H = hankel_matrix(u, 2)
    assert H.shape == (4, 4)
    np.testing.assert_allclose(H[:, 0], [0, 1, 2, 3])
    np.testing.assert_allclose(H[:, -1], [6, 7, 8, 9])


def test_pe_detects_rich_and_poor_inputs():
    rng = np.random.default_rng(0)
    T, L = 200, 10
    t = np.arange(T) * 0.1
    assert persistency_of_excitation([rng.uniform(-1, 1, (T, 4))], L)["persistently_exciting"]
    constant = persistency_of_excitation([np.ones((T, 4))], L)
    assert constant["rank"] == 1 and not constant["persistently_exciting"]
    one_sine = np.sin(np.outer(t, [0.3, 0.5, 0.7, 1.1]) * 2 * np.pi)
    assert persistency_of_excitation([one_sine], L)["rank"] == 8   # 2 per sinusoid


def test_pe_combines_short_episodes_collectively():
    rng = np.random.default_rng(1)
    episodes = [rng.uniform(-1, 1, (12, 2)) for _ in range(10)]   # each too short alone
    single = persistency_of_excitation(episodes[:1], 5)
    combined = persistency_of_excitation(episodes, 5)
    assert not single["persistently_exciting"] and combined["persistently_exciting"]


# --- identification recovers a known linear system ----------------------------

def test_edmd_with_identity_lifting_recovers_hover_model(vehicle):
    model = HoverModel.build(vehicle, np.zeros(3), yaw=0.0, dt=0.1)
    rng = np.random.default_rng(2)
    xs, us, ws = [], [], []
    for _ in range(40):
        x = rng.normal(0, 0.1, 9)
        u = rng.uniform(-0.3, 0.3, (15, 4))
        traj = [x]
        for k in range(14):
            x = model.A @ x + model.B @ u[k]
            traj.append(x)
        xs.append(np.array(traj)); us.append(u); ws.append(np.zeros((15, 3)))
    lifting = SelectiveMonomialLifting(9, 1, nonlinear_indices=[])   # x and 1 only
    g1, g2 = build_snapshots(xs, us, ws, lifting)
    # Winds are all zero, so drop their rows from the regression.
    g1, g2 = g1[:14], g2[:10]
    M = np.linalg.lstsq(g1.T, g2.T, rcond=None)[0].T
    np.testing.assert_allclose(M[:9, :9], model.A, atol=1e-8)
    np.testing.assert_allclose(M[:9, 10:14], model.B, atol=1e-8)


# --- output MPC: the baseline and the Koopman model share one formulation -------

def _hover_output_mpc(model, reference_fn):
    return OutputMPC(model.A, model.B, np.eye(9),
                     lambda x: deviation_state(_outer(x), None, model.x_trim, False)[0],
                     model.u_trim, settings(), reference_fn, model.dt)


def _outer(x13):
    from controllers.linear_mpc import outer_state
    return outer_state(x13)[None]


def test_output_mpc_with_identity_lifting_matches_a_koopman_wrapper(vehicle):
    """A Koopman model whose lifting is the identity must give the baseline's command."""
    model = HoverModel.build(vehicle, np.array([2.7, 1.8, 0.0]), yaw=0.0, dt=0.1)
    reference = lambda t: np.zeros(9)
    baseline = _hover_output_mpc(model, reference)
    scale = np.full(9, 0.5)
    S = np.diag(1.0 / scale)
    # Same linear model in scaled coordinates with a constant feature that stays 1.
    A_l = np.zeros((10, 10)); A_l[:9, :9] = S @ model.A @ np.linalg.inv(S); A_l[9, 9] = 1.0
    B_l = np.zeros((10, 4)); B_l[:9] = S @ model.B
    koop = OuterKoopmanModel(A=A_l, B=B_l, D=np.zeros((10, 3)), scale=scale, degree=1,
                             nonlinear_indices=(), include_rates=False, x_trim=model.x_trim,
                             u_trim=model.u_trim, wind_mean=model.wind_mean, dt=0.1)
    koopman_mpc = OutputMPC(koop.A, koop.B, koop.C, koop.lift_measurement, koop.u_trim,
                            settings(), reference, koop.dt)
    x = full_state(model.x_trim + np.array([0.3, -0.2, 0.1, 0.1, 0, 0, 0.02, 0, 0.05]))
    np.testing.assert_allclose(koopman_mpc(0.0, x), baseline(0.0, x), atol=1e-8)


def test_output_mpc_tracks_a_circle_reference_in_prediction(vehicle):
    model = HoverModel.build(vehicle, np.zeros(3), yaw=0.0, dt=0.1)
    reference = lambda t: np.array([0.5, 0, 0, 0, 0, 0, 0, 0, 0])
    mpc = _hover_output_mpc(model, reference)
    u = mpc(0.0, full_state(model.x_trim))
    # Target 0.5 m north: the first move pitches nose down (negative pitch rate).
    assert u[1] < 0
    np.testing.assert_allclose(mpc.log.error[0][0], -0.5)


def test_horizon_rmse_is_zero_for_the_true_model(vehicle):
    model = HoverModel.build(vehicle, np.zeros(3), yaw=0.0, dt=0.1)
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.1, 9); u = rng.uniform(-0.3, 0.3, (30, 4)); xs = [x]
    for k in range(29):
        x = model.A @ x + model.B @ u[k]; xs.append(x)
    ep = {"dx": np.array(xs), "y": np.array(xs), "du": u, "dw": np.zeros((30, 3))}
    err = horizon_rmse(model.A, model.B, model.D, np.eye(9), lambda d: d, [ep], 10)
    assert err.shape == (10, 9) and np.max(err) < 1e-12


def test_model_round_trips_to_disk(tmp_path):
    m = OuterKoopmanModel(A=np.eye(13), B=np.zeros((13, 4)), D=np.zeros((13, 3)),
                          scale=np.ones(12), degree=1, nonlinear_indices=(3, 4),
                          include_rates=True, x_trim=np.zeros(9), u_trim=np.array([0, 0, 0, 9.81]),
                          wind_mean=np.zeros(3), dt=0.1)
    m.save(tmp_path)
    loaded = OuterKoopmanModel.load(tmp_path)
    assert loaded.nonlinear_indices == (3, 4) and loaded.include_rates
    np.testing.assert_allclose(loaded.C, m.C)


# --- data collection ------------------------------------------------------------

def test_dataset_round_trips_without_pickle(tmp_path):
    from datagen.outer_loop import Episode, load_episodes, save_episodes
    rng = np.random.default_rng(4)
    make = lambda name, T: Episode(name, rng.normal(size=(T, 9)), rng.normal(size=(T, 3)),
                                   rng.normal(size=(T, 4)), rng.normal(size=(T, 3)))
    splits = {"train": [make("a", 5), make("b", 7)], "validation": [make("c", 4)]}
    save_episodes(tmp_path, splits)
    loaded = load_episodes(tmp_path)
    assert [ep.name for ep in loaded["train"]] == ["a", "b"]
    np.testing.assert_allclose(loaded["train"][1].u, splits["train"][1].u)
    np.testing.assert_allclose(loaded["validation"][0].x9, splits["validation"][0].x9)


def test_dither_stays_within_the_command_bounds():
    from datagen.outer_loop import DitheredController

    class Saturated:
        period = 0.1

        def __call__(self, t, x):
            return np.array([RATE, -RATE, 0.0, 18.0])   # already on the bounds

    lb, ub = np.array([-RATE] * 3 + [2.0]), np.array([RATE] * 3 + [18.0])
    dithered = DitheredController(Saturated(), [0.3, 0.3, 0.3, 1.0], lb, ub,
                                  np.random.default_rng(5))
    commands = np.array([dithered(0.1 * k, None) for k in range(200)])
    assert np.all(commands >= lb) and np.all(commands <= ub)
    assert np.std(commands[:, 2]) > 0.1   # the free channel is actually dithered


# --- translation-invariant fit ----------------------------------------------------

def test_invariant_fit_ignores_a_position_that_never_varied(vehicle):
    """All data at one altitude: the plain fit can tie p_z to the bias; the invariant one cannot."""
    from koopman.structured_edmd import fit_translation_invariant
    model = HoverModel.build(vehicle, np.zeros(3), yaw=0.0, dt=0.1)
    rng = np.random.default_rng(6)
    xs, us, ws = [], [], []
    for _ in range(40):
        x = rng.normal(0, 0.1, 9)
        x[2] = -2.0                                    # same altitude everywhere
        u = rng.uniform(-0.3, 0.3, (15, 4)); traj = [x]
        for k in range(14):
            x = model.A @ x + model.B @ u[k]; traj.append(x)
        xs.append(np.array(traj)); us.append(u); ws.append(rng.normal(0, 0.1, (15, 3)))
    lifting = SelectiveMonomialLifting(9, 1, nonlinear_indices=[])
    g1, g2 = build_snapshots(xs, us, ws, lifting)
    A, B, D = fit_translation_invariant(g1, g2, 10, 4, 3, [0, 1, 2], lifting.constant_index)
    np.testing.assert_allclose(A[:, :3], np.eye(10)[:, :3])          # p feeds only itself
    np.testing.assert_allclose(A[:9, 3:9], model.A[:, 3:9], atol=1e-6)
    np.testing.assert_allclose(B[:9], model.B, atol=1e-6)
    np.testing.assert_allclose(A[9], np.eye(10)[9])                  # constant row fixed


def _gap_closure_script():
    import importlib.util
    from conftest import REPO_ROOT
    spec = importlib.util.spec_from_file_location(
        "compare_gap_closure", REPO_ROOT / "scripts" / "compare_gap_closure.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def _model_about(hover: HoverModel) -> OuterKoopmanModel:
    """A placeholder lifted model whose coordinates are deviations from `hover`'s trim."""
    n = 9 + 1
    return OuterKoopmanModel(A=np.eye(n), B=np.zeros((n, 4)), D=np.zeros((n, 3)),
                             scale=np.ones(9), degree=1, nonlinear_indices=(),
                             include_rates=False, x_trim=hover.x_trim, u_trim=hover.u_trim,
                             wind_mean=hover.wind_mean, dt=hover.dt)


def test_gap_closure_accepts_only_a_model_in_nominal_coordinates(vehicle):
    from conftest import REPO_ROOT
    from experiment.ltv_mpc_config import load_ltv_mpc
    script = _gap_closure_script()
    configs = REPO_ROOT / "configs" / "controllers"
    nominal = load_ltv_mpc(configs / "ltv_mpc_nominal.toml")
    oracle = load_ltv_mpc(configs / "ltv_mpc.toml")
    script.check_compatible(_model_about(nominal.hover_model(vehicle)), nominal, oracle, vehicle)
    with pytest.raises(ValueError, match="nominal hover trim"):
        script.check_compatible(_model_about(oracle.hover_model(vehicle)), nominal, oracle,
                                vehicle)
    with pytest.raises(ValueError, match="drag scale"):
        script.check_compatible(_model_about(nominal.hover_model(vehicle)), oracle, oracle,
                                vehicle)


def test_trig_model_round_trips_and_old_models_load_without_trig(tmp_path, vehicle):
    import json
    hover = HoverModel.build(vehicle, np.zeros(3), 0.0, 0.1)
    lifting_dim = TrigLifting(12, 2, [3, 4, 5], [6, 7, 8], [0.1, 0.1, 0.1]).lifted_dim
    model = OuterKoopmanModel(A=np.eye(lifting_dim), B=np.zeros((lifting_dim, 4)),
                              D=np.zeros((lifting_dim, 3)), scale=np.full(12, 0.1), degree=2,
                              nonlinear_indices=(3, 4, 5), include_rates=True,
                              x_trim=hover.x_trim, u_trim=hover.u_trim,
                              wind_mean=np.zeros(3), dt=0.1, trig_indices=(6, 7, 8))
    model.save(tmp_path)
    loaded = OuterKoopmanModel.load(tmp_path)
    assert loaded.trig_indices == (6, 7, 8)
    dx = np.random.default_rng(0).normal(scale=0.1, size=(5, 12))
    np.testing.assert_array_equal(loaded.lift_deviation(dx), model.lift_deviation(dx))

    spec = json.loads((tmp_path / "outer_model.json").read_text())
    del spec["trig_indices"]                       # a run saved before trig features existed
    (tmp_path / "outer_model.json").write_text(json.dumps(spec))
    assert OuterKoopmanModel.load(tmp_path).trig_indices == ()
