"""K-MPC config loading, the simulator adapter, its log, and the run_kmpc script.

A synthetic identification run (degree-1 lifting of the 9-dim error, so N = 10)
is written to tmp_path, so these tests do not depend on results/.
"""
import importlib.util
import json

import numpy as np
import pytest

from conftest import REPO_ROOT
from datagen.collect import tracking_error
from experiment.augmentation import KMPCAugmentation, MPCLog
from experiment.builders import load_experiment, load_kmpc, load_toml
from experiment.run_record import load_log
from koopman.models import KoopmanModel
from simulation.closed_loop import Timing, simulate

BASELINE = REPO_ROOT / "configs" / "baseline.toml"
KMPC_CONFIG = REPO_ROOT / "configs" / "controllers" / "kmpc.toml"
N_ERR, N_LIFT, N_IN, N_WIND = 9, 10, 4, 3


def write_model_run(tmp_path, degree=1):
    """model.npz, standardizer.npz and identification.json of a stable synthetic model."""
    rng = np.random.default_rng(0)
    run = tmp_path / "model_run"
    run.mkdir()
    A = 0.8 * np.eye(N_LIFT)
    A[N_ERR, N_ERR] = 1.0  # constant monomial
    B = 0.05 * rng.standard_normal((N_LIFT, N_IN))
    D = 0.05 * rng.standard_normal((N_LIFT, N_WIND))
    B[N_ERR] = D[N_ERR] = 0.0
    KoopmanModel(A=A, B=B, D=D, C=np.eye(N_ERR, N_LIFT)).save(run / "model.npz")
    np.savez(run / "standardizer.npz", mean=np.full(N_ERR, 0.01), std=np.full(N_ERR, 0.1))
    (run / "identification.json").write_text(json.dumps({"state_dim": N_ERR, "degree": degree}))
    return run


def write_kmpc(tmp_path, model_run, **overrides):
    values = {
        "model_run": f'"{model_run}"',
        "sample_time": "0.1",
        "horizon": "5",
        "Q_diag": str([100.0] * 3 + [10.0] * 3 + [1.0] * 3),
        "R_diag": "[1.0, 5.0, 5.0, 0.1]",
        "u_bound": "[0.3, 0.3, 0.3, 1.0]",
        "h_lb": "[" + ", ".join(["-inf"] * N_ERR) + "]",
        "h_ub": "[" + ", ".join(["inf"] * 8 + ["0.5"]) + "]",
        "wind_mean": "[1.0, 0.0, 0.0]",
    }
    values.update(overrides)
    path = tmp_path / "kmpc.toml"
    path.write_text("[kmpc]\n" + "".join(f"{k} = {v}\n" for k, v in values.items() if v is not None))
    return path


def write_experiment(tmp_path, duration=2.0):
    """Hover in a constant 1 m/s north wind, pointing at the repo's real configs."""
    path = tmp_path / "hover.toml"
    path.write_text(f"""
[experiment]
name = "hover"
seed = 3
duration = {duration}
vehicle_config = "{REPO_ROOT / 'configs' / 'quadrotor.toml'}"
controller_config = "{BASELINE}"

[initial_state]
position = [0.0, 0.0, -1.0]
euler_deg = [0.0, 0.0, 0.0]
velocity = [0.0, 0.0, 0.0]
body_rate = [0.0, 0.0, 0.0]

[reference]
type = "setpoint"
position = [0.0, 0.0, -1.0]
yaw_deg = 0.0

[wind]
type = "constant"
velocity = [1.0, 0.0, 0.0]
""")
    return path


# --- config ------------------------------------------------------------------

def test_load_kmpc_builds_settings_in_physical_units(tmp_path):
    cfg = load_kmpc(write_kmpc(tmp_path, write_model_run(tmp_path)))
    s = cfg.settings
    np.testing.assert_array_equal(np.diag(s.Q), [100.0] * 3 + [10.0] * 3 + [1.0] * 3)
    np.testing.assert_array_equal(s.u_lb, -s.u_ub)
    np.testing.assert_array_equal(s.u_ub, [0.3, 0.3, 0.3, 1.0])
    assert np.all(np.isneginf(s.h_lb))
    assert s.h_ub[-1] == 0.5 and np.all(np.isposinf(s.h_ub[:-1]))
    assert s.horizon == 5 and cfg.sample_time == 0.1
    assert cfg.lifting.lifted_dim == N_LIFT


def test_make_controller_returns_independent_controllers(tmp_path):
    cfg = load_kmpc(write_kmpc(tmp_path, write_model_run(tmp_path)))
    assert cfg.make_controller() is not cfg.make_controller()


@pytest.mark.parametrize("overrides, match", [
    ({"horizon": None}, "missing key 'horizon'"),
    ({"Q_diag": "[1.0, 2.0]"}, "Q_diag must be 9"),
    ({"u_bound": "[0.3, -0.3, 0.3, 1.0]"}, "non-negative"),
    ({"h_ub": "[" + ", ".join(["nan"] * N_ERR) + "]"}, "h_ub"),
    ({"horizon": "0"}, "positive integer"),
    ({"sample_time": "-0.1"}, "sample_time"),
])
def test_load_kmpc_rejects_bad_config(tmp_path, overrides, match):
    path = write_kmpc(tmp_path, write_model_run(tmp_path), **overrides)
    with pytest.raises(ValueError, match=match):
        load_kmpc(path)


def test_load_kmpc_rejects_missing_model_run(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        load_kmpc(write_kmpc(tmp_path, tmp_path / "nowhere"))


def test_load_kmpc_rejects_lifting_that_does_not_match_the_model(tmp_path):
    # identification.json says degree 2 (N = 55), model.npz has N = 10.
    with pytest.raises(ValueError, match="lifting maps"):
        load_kmpc(write_kmpc(tmp_path, write_model_run(tmp_path, degree=2)))


def test_check_timing_requires_dt_augment_equal_to_sample_time(tmp_path):
    cfg = load_kmpc(write_kmpc(tmp_path, write_model_run(tmp_path)))
    cfg.check_timing(Timing())
    with pytest.raises(ValueError, match="sample_time"):
        cfg.check_timing(Timing(dt_gain=0.2, dt_augment=0.2, dt_log=0.2))


@pytest.mark.skipif(not KMPC_CONFIG.exists(), reason="no K-MPC config")
def test_repo_kmpc_config_loads_when_its_model_run_exists():
    model_run = (KMPC_CONFIG.parent / load_toml(KMPC_CONFIG)["kmpc"]["model_run"]).resolve()
    if not model_run.is_dir():
        pytest.skip(f"model run {model_run.name} not present (results/ is not tracked)")
    cfg = load_kmpc(KMPC_CONFIG)
    assert cfg.lifting.lifted_dim == cfg.model.A.shape[0]


# --- adapter and log ---------------------------------------------------------

def test_augmentation_solves_on_the_tracking_error(tmp_path):
    cfg = load_kmpc(write_kmpc(tmp_path, write_model_run(tmp_path)))
    exp = load_experiment(write_experiment(tmp_path))
    x = exp.x0.copy()
    x[0] += 0.2  # 0.2 m north of the reference
    x_ref, _ = exp.reference(0.0)

    augmentation = KMPCAugmentation(cfg.make_controller())
    u_d = augmentation(0.0, x, x_ref)
    e_x = tracking_error(x[None, :], x_ref[None, :])[0]
    np.testing.assert_allclose(augmentation.log().e_x[0], e_x)
    np.testing.assert_allclose(u_d, cfg.make_controller().solve(e_x).u_d, atol=1e-12)


def test_mpc_log_round_trips(tmp_path):
    cfg = load_kmpc(write_kmpc(tmp_path, write_model_run(tmp_path)))
    exp = load_experiment(write_experiment(tmp_path))
    x_ref, _ = exp.reference(0.0)
    augmentation = KMPCAugmentation(cfg.make_controller())
    for t in (0.0, 0.1):
        augmentation(t, exp.x0, x_ref)
    log = augmentation.log()
    log.save(tmp_path / "mpc_log.npz")
    loaded = MPCLog.load(tmp_path / "mpc_log.npz")
    for name in ("t", "e_x", "u_d", "solved", "planned_inputs", "predicted_error"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(log, name))
    assert loaded.planned_inputs.shape == (2, 5, N_IN)
    assert loaded.predicted_error.shape == (2, 5, N_ERR)


def test_empty_log_raises(tmp_path):
    cfg = load_kmpc(write_kmpc(tmp_path, write_model_run(tmp_path)))
    with pytest.raises(ValueError, match="no K-MPC solves"):
        KMPCAugmentation(cfg.make_controller()).log()


def test_closed_loop_smoke_solves_every_augment_step(tmp_path):
    cfg = load_kmpc(write_kmpc(tmp_path, write_model_run(tmp_path)))
    exp = load_experiment(write_experiment(tmp_path, duration=2.0))
    cfg.check_timing(exp.timing)
    augmentation = KMPCAugmentation(cfg.make_controller())
    log = simulate(exp.x0, exp.duration, exp.reference, exp.wind, exp.params,
                   exp.lqr, exp.rate_controller, exp.timing, augmentation=augmentation)
    mpc_log = augmentation.log()

    np.testing.assert_allclose(mpc_log.t, np.arange(20) * 0.1, atol=1e-12)
    assert augmentation.fallback_count == 0
    assert np.all(np.isfinite(log.x))
    # dt_log == dt_augment: the logged u_d is the one each solve returned.
    np.testing.assert_allclose(log.u_d, mpc_log.u_d)
    assert np.all(np.abs(mpc_log.u_d) <= cfg.settings.u_ub + 1e-9)


# --- script ------------------------------------------------------------------

def test_run_kmpc_script_writes_the_run_record(tmp_path):
    spec = importlib.util.spec_from_file_location("run_kmpc", REPO_ROOT / "scripts" / "run_kmpc.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    kmpc_path = write_kmpc(tmp_path, write_model_run(tmp_path))
    run_dir = script.main([str(write_experiment(tmp_path, duration=1.0)), str(kmpc_path),
                           "--results-dir", str(tmp_path / "runs"), "--no-show"])

    assert run_dir.name.endswith("_hover_kmpc")
    assert (run_dir / "config" / "kmpc.toml").exists()
    assert json.loads((run_dir / "kmpc_identification.json").read_text())["degree"] == 1
    assert len(load_log(run_dir).t) == 10
    assert len(MPCLog.load(run_dir / "mpc_log.npz").t) == 10
    assert (run_dir / "figures" / "tracking.png").exists()
