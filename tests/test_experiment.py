"""Config builders, setpoint reference, run records and the baseline script."""
from datetime import datetime, timezone
import importlib.util
import json

import numpy as np
import pytest

from conftest import REPO_ROOT
from controllers import sdlqr_model as lqr_model
from datagen.trajectories import SetpointReference
from dynamics import dynamics as rigid_body
from experiment.builders import (
    build_lqr,
    build_timing,
    build_wind,
    load_experiment,
    load_toml,
)
from experiment.run_record import git_state, load_log, save_log, start_run
from simulation.closed_loop import lqr_state as lqr_state_of
from simulation.closed_loop import simulate

BASELINE = REPO_ROOT / "configs" / "baseline.toml"
HOVER_WIND = REPO_ROOT / "configs" / "experiments" / "hover_wind.toml"
FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def write_experiment(tmp_path, duration=1.0, wind="constant"):
    """Short copy of hover_wind.toml pointing at the repo's real configs."""
    path = tmp_path / "short.toml"
    path.write_text(f"""
[experiment]
name = "short"
seed = 7
duration = {duration}
vehicle_config = "{REPO_ROOT / 'configs' / 'quadrotor.toml'}"
controller_config = "{BASELINE}"

[initial_state]
position = [0.0, 0.0, 0.0]
euler_deg = [0.0, 0.0, 90.0]
velocity = [0.0, 0.0, 0.0]
body_rate = [0.0, 0.0, 0.0]

[reference]
type = "setpoint"
position = [0.0, 0.0, -1.0]
yaw_deg = 90.0

[wind]
type = "{wind}"
velocity = [1.0, 0.0, 0.0]
""")
    return path


def test_baseline_config_builds_matching_objects():
    cfg = load_toml(BASELINE)
    timing = build_timing(cfg)
    lqr = build_lqr(cfg, timing)
    np.testing.assert_array_equal(np.diag(lqr.Q), cfg["lqr"]["Q_diag"])
    np.testing.assert_array_equal(np.diag(lqr.R), cfg["lqr"]["R_diag"])
    assert lqr.dt == timing.dt_control == cfg["timing"]["dt_control"]
    assert lqr.thrust_limits == tuple(cfg["lqr"]["thrust_limits"])


def test_wrong_length_weight_raises_with_location():
    cfg = load_toml(BASELINE)
    cfg["lqr"]["Q_diag"] = cfg["lqr"]["Q_diag"][:9]
    with pytest.raises(ValueError, match=r"\[lqr\] Q_diag"):
        build_lqr(cfg, build_timing(cfg))


def test_unknown_wind_type_raises():
    with pytest.raises(ValueError, match="unknown wind type"):
        build_wind({"wind": {"type": "hurricane"}}, np.random.default_rng(0))


def test_missing_section_raises():
    with pytest.raises(ValueError, match=r"\[timing\]"):
        build_timing({})


def test_shipped_experiment_loads():
    exp = load_experiment(HOVER_WIND)
    assert exp.name == "hover_wind"
    assert exp.duration == 15.0
    rotor_files = [] if exp.actuator.model == "ideal" else ["rotor.toml"]   # per [rotor] model
    assert [p.name for p in exp.config_paths] == ["hover_wind.toml", "quadrotor.toml",
                                                  "baseline.toml", *rotor_files]


def test_experiment_initial_state_and_reference(tmp_path):
    exp = load_experiment(write_experiment(tmp_path))
    yaw_90 = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    np.testing.assert_allclose(exp.x0[rigid_body.QUAT], yaw_90, atol=1e-12)
    x_ref, _ = exp.reference(0.0)
    np.testing.assert_allclose(x_ref[lqr_model.QUAT], yaw_90, atol=1e-12)


def test_setpoint_reference_is_level_hover():
    x_ref, u_ref = SetpointReference([1.0, 2.0, -3.0], 0.0, 9.81)(5.0)
    np.testing.assert_array_equal(x_ref[lqr_model.POS], [1.0, 2.0, -3.0])
    np.testing.assert_allclose(x_ref[lqr_model.QUAT], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(x_ref[lqr_model.VEL], 0.0)
    np.testing.assert_array_equal(u_ref, [0.0, 0.0, 0.0, 9.81])


def test_setpoint_reference_returns_copies():
    reference = SetpointReference(np.zeros(3), 0.0, 9.81)
    x_ref, _ = reference(0.0)
    x_ref[:] = np.nan
    assert np.all(np.isfinite(reference(0.0)[0]))


def test_git_state_reports_commit():
    state = git_state(REPO_ROOT)
    assert len(state["commit"]) == 40
    assert isinstance(state["dirty"], bool)


def test_git_state_outside_repo_raises(tmp_path):
    with pytest.raises(RuntimeError):
        git_state(tmp_path)


def test_start_run_writes_metadata_and_verbatim_configs(tmp_path):
    exp = load_experiment(write_experiment(tmp_path))
    run_dir = start_run(tmp_path / "runs", exp.name, exp.seed, exp.config_paths,
                        REPO_ROOT, now=FIXED_TIME)

    assert run_dir.name == "20260102T030405Z_short"
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["seed"] == 7
    assert metadata["git_commit"] == git_state(REPO_ROOT)["commit"]
    assert (run_dir / "git_diff.patch").exists() == bool(git_state(REPO_ROOT)["diff"])
    for path in exp.config_paths:
        assert (run_dir / "config" / path.name).read_bytes() == path.read_bytes()


def test_start_run_records_non_utf8_diff_verbatim(tmp_path):
    # Regression: a modified tracked file with non-UTF-8 bytes that git diffs as
    # text (as a PDF can be) made start_run raise UnicodeDecodeError.
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True,
                                    capture_output=True)
    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    blob = repo / "figure.pdf"
    blob.write_bytes(b"%PDF-1.7\n\x80\x81 first\n")
    git("add", "figure.pdf")
    git("commit", "-qm", "init")
    blob.write_bytes(b"%PDF-1.7\n\x80\x82 second\n")

    run_dir = start_run(tmp_path / "runs", "binary_diff", 0, [], repo)
    patch = (run_dir / "git_diff.patch").read_bytes()
    assert b"\x80\x82 second" in patch and b"\x80\x81 first" in patch


def test_start_run_refuses_to_overwrite(tmp_path):
    exp = load_experiment(write_experiment(tmp_path))
    args = (tmp_path / "runs", exp.name, exp.seed, exp.config_paths, REPO_ROOT)
    start_run(*args, now=FIXED_TIME)
    with pytest.raises(FileExistsError):
        start_run(*args, now=FIXED_TIME)


def test_log_round_trips_exactly(tmp_path):
    exp = load_experiment(write_experiment(tmp_path))
    log = simulate(exp.x0, exp.duration, exp.reference, exp.wind, exp.params,
                   exp.lqr, exp.rate_controller, exp.timing)
    save_log(tmp_path, log)
    loaded = load_log(tmp_path)

    assert loaded.timing == log.timing
    for name in ("t", "x", "x_ref", "u_ref", "u_h", "u_d", "u_cmd", "v_w"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(log, name))


def test_baseline_script_runs_end_to_end(tmp_path):
    spec = importlib.util.spec_from_file_location("run_baseline", REPO_ROOT / "scripts" / "run_baseline.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    run_dir = script.main([str(write_experiment(tmp_path)), "--results-dir", str(tmp_path / "runs"),
                           "--no-show"])
    assert (run_dir / "metadata.json").exists()
    assert len(load_log(run_dir).t) == 10
    assert (run_dir / "figures" / "tracking.png").exists()
    assert (run_dir / "figures" / "trajectory_3d.png").exists()


def test_plot_run_script_replots_saved_run(tmp_path):
    exp = load_experiment(write_experiment(tmp_path))
    log = simulate(exp.x0, exp.duration, exp.reference, exp.wind, exp.params,
                   exp.lqr, exp.rate_controller, exp.timing)
    save_log(tmp_path, log)

    spec = importlib.util.spec_from_file_location("plot_run", REPO_ROOT / "scripts" / "plot_run.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    paths = script.main([str(tmp_path), "--no-show"])
    assert sorted(p.name for p in paths) == ["tracking.png", "trajectory_3d.png"]


def test_figure8_experiment_starts_on_reference():
    exp = load_experiment(REPO_ROOT / "configs" / "experiments" / "figure8_wind.toml")
    x_ref, u_ref = exp.reference(0.0)
    np.testing.assert_allclose(lqr_state_of(exp.x0), x_ref)
    np.testing.assert_allclose(exp.x0[rigid_body.RATE], u_ref[lqr_model.RATE])
