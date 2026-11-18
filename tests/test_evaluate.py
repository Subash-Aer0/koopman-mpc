"""Paired Monte Carlo, evaluation config, and the 03_evaluate_controllers script."""
import importlib.util
import json

import numpy as np
import pytest

from conftest import REPO_ROOT
from experiment.builders import load_evaluation, load_experiment
from experiment.run_record import load_log
from simulation.monte_carlo import run_paired
from test_kmpc_experiment import write_kmpc, write_model_run

BASELINE = REPO_ROOT / "configs" / "baseline.toml"


def write_gaussian_experiment(tmp_path, name="hover_gust", duration=0.5):
    path = tmp_path / f"{name}.toml"
    path.write_text(f"""
[experiment]
name = "{name}"
seed = 0
duration = {duration}
vehicle_config = "{REPO_ROOT / 'configs' / 'quadrotor.toml'}"
controller_config = "{BASELINE}"

[initial_state]
start_on_reference = true

[reference]
type = "setpoint"
position = [0.0, 0.0, -1.0]
yaw_deg = 0.0

[wind]
type = "gaussian"
mean = [2.0, 1.0, 0.0]
cov = [[0.5, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.01]]
hold_period = 0.1
""")
    return path


class CountingAugmentation:
    """Constant small u_d; counts calls, so shared instances would show."""

    def __init__(self):
        self.calls = 0

    def __call__(self, t, x, x_ref):
        self.calls += 1
        return np.array([0.0, 0.0, 0.0, 0.2])


# --- paired Monte Carlo ------------------------------------------------------

def test_controllers_on_one_seed_fly_the_same_wind(tmp_path):
    path = write_gaussian_experiment(tmp_path)
    results = run_paired(lambda seed: load_experiment(path, seed),
                         {"lqr": lambda: None, "other": CountingAugmentation}, [11, 12])
    for i in range(2):
        np.testing.assert_array_equal(results["lqr"][i].log.v_w, results["other"][i].log.v_w)
    assert not np.array_equal(results["lqr"][0].log.v_w, results["lqr"][1].log.v_w)
    assert [r.seed for r in results["other"]] == [11, 12]


def test_identical_controllers_give_identical_runs(tmp_path):
    # Fresh LQR and wind per run: no state leaks from one run into the next.
    path = write_gaussian_experiment(tmp_path)
    results = run_paired(lambda seed: load_experiment(path, seed),
                         {"a": lambda: None, "b": lambda: None}, [5, 6])
    for ra, rb in zip(results["a"], results["b"]):
        np.testing.assert_array_equal(ra.log.x, rb.log.x)


def test_each_run_gets_fresh_episode_and_augmentation(tmp_path):
    path = write_gaussian_experiment(tmp_path)
    built = []

    def make_episode(seed):
        built.append(seed)
        return load_experiment(path, seed)

    results = run_paired(make_episode, {"a": CountingAugmentation, "b": CountingAugmentation},
                         [1, 2, 3])
    assert built == [1, 1, 2, 2, 3, 3]
    augmentations = [r.augmentation for rs in results.values() for r in rs]
    assert len({id(a) for a in augmentations}) == 6
    assert all(a.calls == 5 for a in augmentations)  # 0.5 s at 10 Hz


@pytest.mark.parametrize("controllers, seeds, match", [
    ({}, [1], "no controllers"),
    ({"a": lambda: None}, [], "no seeds"),
    ({"a": lambda: None}, [1, 1], "distinct"),
])
def test_run_paired_rejects_bad_arguments(tmp_path, controllers, seeds, match):
    path = write_gaussian_experiment(tmp_path)
    with pytest.raises(ValueError, match=match):
        run_paired(lambda seed: load_experiment(path, seed), controllers, seeds)


def test_load_experiment_seed_override(tmp_path):
    path = write_gaussian_experiment(tmp_path)
    assert load_experiment(path).seed == 0
    assert load_experiment(path, seed=42).seed == 42
    with pytest.raises(ValueError, match="non-negative"):
        load_experiment(path, seed=-1)


# --- evaluation config -------------------------------------------------------

def write_evaluation(tmp_path, n_runs=2, experiments=None):
    model_run = write_model_run(tmp_path)
    kmpc = write_kmpc(tmp_path, model_run)
    experiments = experiments or [write_gaussian_experiment(tmp_path, "gust_a"),
                                  write_gaussian_experiment(tmp_path, "gust_b")]
    path = tmp_path / "evaluate.toml"
    path.write_text(f"""
[evaluate]
name = "eval"
seed = 2026
n_runs = {n_runs}
experiments = [{", ".join(f'"{p}"' for p in experiments)}]
kmpc_config = "{kmpc}"
""")
    return path


def test_load_evaluation_and_run_seeds(tmp_path):
    cfg = load_evaluation(write_evaluation(tmp_path, n_runs=4))
    seeds = cfg.run_seeds()
    assert len(seeds) == 4 and len(set(seeds)) == 4
    assert seeds == cfg.run_seeds()  # reproducible from the root seed
    assert [p.stem for p in cfg.experiments] == ["gust_a", "gust_b"]


@pytest.mark.parametrize("n_runs", ["1", "true", "2.0"])  # TOML literals
def test_load_evaluation_rejects_bad_n_runs(tmp_path, n_runs):
    with pytest.raises(ValueError, match="≥ 2"):
        load_evaluation(write_evaluation(tmp_path, n_runs=n_runs))


def test_load_evaluation_rejects_duplicate_experiment_names(tmp_path):
    first = write_gaussian_experiment(tmp_path, "same")
    second_dir = tmp_path / "other"
    second_dir.mkdir()
    second = write_gaussian_experiment(second_dir, "same")
    with pytest.raises(ValueError, match="distinct"):
        load_evaluation(write_evaluation(tmp_path, experiments=[first, second]))


def test_repo_evaluation_config_loads():
    cfg = load_evaluation(REPO_ROOT / "configs" / "evaluate" / "kmpc_vs_lqr.toml")
    assert cfg.n_runs >= 2
    assert all(p.exists() for p in cfg.experiments + [cfg.kmpc])


# --- script ------------------------------------------------------------------

def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_evaluate_script_writes_paired_results(tmp_path):
    script = load_script("03_evaluate_controllers")
    run_dir = script.main([str(write_evaluation(tmp_path)), "--results-dir", str(tmp_path / "runs")])

    summary = json.loads((run_dir / "summary.json").read_text())
    assert set(summary["experiments"]) == {"gust_a", "gust_b"}
    exp = summary["experiments"]["gust_a"]
    assert len(exp["per_run"]["lqr"]) == len(exp["per_run"]["kmpc"]) == 2
    assert exp["comparison"]["euclidean_norm"]["n"] == 2
    assert exp["kmpc_solver"]["solves"] == 10  # 2 runs x 0.5 s at 10 Hz

    run_00 = run_dir / "episodes" / "gust_a" / "kmpc" / "run_00"
    assert len(load_log(run_00).t) == 5 and (run_00 / "mpc_log.npz").exists()
    assert not (run_dir / "episodes" / "gust_a" / "lqr" / "run_00" / "mpc_log.npz").exists()
    with np.load(run_dir / "position_errors.npz") as errors:
        assert errors["gust_b/kmpc"].shape == (2, 5, 3)
    assert "gust_b (2 paired runs)" in (run_dir / "summary.md").read_text()
    # Shared vehicle and controller configs are copied once (rotor.toml when the vehicle uses it).
    from dynamics.actuator import load_actuator
    from experiment.builders import load_toml
    vehicle = REPO_ROOT / "configs" / "quadrotor.toml"
    rotor_files = [p.name for p in load_actuator(load_toml(vehicle), vehicle).config_paths]
    assert sorted(p.name for p in (run_dir / "config").iterdir()) == sorted(
        ["evaluate.toml", "kmpc.toml", "gust_a.toml", "gust_b.toml",
         "quadrotor.toml", "baseline.toml", *rotor_files])


def test_plot_evaluation_script_plots_every_experiment(tmp_path):
    run_dir = load_script("03_evaluate_controllers").main(
        [str(write_evaluation(tmp_path)), "--results-dir", str(tmp_path / "runs")])
    paths = load_script("plot_evaluation").main([str(run_dir), "--no-show"])
    assert sorted(p.name for p in paths) == ["gust_a_abs_error.png", "gust_a_mse.png",
                                             "gust_b_abs_error.png", "gust_b_mse.png"]
    assert all(p.parent == run_dir / "figures" and p.stat().st_size > 0 for p in paths)
