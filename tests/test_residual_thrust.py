"""scripts/check_residual_thrust.py: f_res is zero with an ideal motor, non-zero with a motor."""
import importlib.util
import json

import numpy as np

from conftest import REPO_ROOT
from experiment.builders import load_experiment


def load_script():
    spec = importlib.util.spec_from_file_location(
        "check_residual_thrust", REPO_ROOT / "scripts" / "check_residual_thrust.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def short_experiment(tmp_path):
    """The paper's figure-8, shortened to 1 s, vehicle path absolute."""
    text = (REPO_ROOT / "configs/experiments/figure8_gaussian.toml").read_text()
    lines = []
    for line in text.splitlines():
        key = line.split("=")[0].strip()
        if key == "duration":
            line = "duration = 1.0"
        elif key in ("vehicle_config", "controller_config"):
            rel = line.split("=", 1)[1].split("#")[0].strip().strip('"')
            line = f'{key} = "{(REPO_ROOT / "configs/experiments" / rel).resolve()}"'
        lines.append(line)
    path = tmp_path / "figure8_short.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_vehicle_override_keeps_the_wind_and_swaps_the_actuator(tmp_path):
    path = short_experiment(tmp_path)
    default = load_experiment(path, seed=1)
    motor = load_experiment(path, seed=1, vehicle_config=REPO_ROOT / "configs/quadrotor_motor.toml")
    assert default.actuator.motor is None and motor.actuator.motor is not None
    np.testing.assert_array_equal(default.wind.sample(0.35, np.zeros(3)),
                                  motor.wind.sample(0.35, np.zeros(3)))


def test_residual_is_exactly_zero_without_a_motor_and_not_with_one(tmp_path):
    script = load_script()
    path = short_experiment(tmp_path)
    reports = {}
    for name in ("quadrotor.toml", "quadrotor_motor.toml"):
        run_dir = script.main([str(path), "--vehicle", str(REPO_ROOT / "configs" / name),
                               "--seeds", "1", "--results-dir", str(tmp_path / name)])
        reports[name] = json.loads((run_dir / "summary.json").read_text())["pooled"]
    assert reports["quadrotor.toml"]["rms_c_res_m_s2"] < 1e-12       # Ω_R = Ω_cmd
    assert reports["quadrotor.toml"]["rms_c_miss_m_s2"] > 0          # BEMT still misses
    assert reports["quadrotor_motor.toml"]["rms_c_res_m_s2"] > 1e-6
