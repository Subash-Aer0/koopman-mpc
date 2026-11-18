"""Data generation: perturbations, Gaussian wind, circle reference, tracking error, datasets."""
import importlib.util

import numpy as np
import pytest

from conftest import REPO_ROOT
from controllers.sdlqr_model import POS, nominal_dynamics
from datagen.collect import (
    ERR_POS,
    ERR_RPY,
    ERR_VEL,
    HeldPerturbations,
    random_perturbations,
    tracking_error,
)
from datagen.dataset import load_dataset, save_dataset
from datagen.trajectories import CircleReference, SumOfSinesReference, sample_sum_of_sines
from dynamics import dynamics as rigid_body
from dynamics.quaternion import Quaternion
from experiment.builders import load_datagen, state_on_reference
from wind.wind_model import GaussianWind

G = 9.81
BOUNDS = np.array([0.3, 0.3, 0.3, 1.0])
WIND_MEAN = np.array([2.0, 1.0, 0.0])
WIND_COV = np.diag([0.5, 0.4, 0.01])
MODEL1 = REPO_ROOT / "configs" / "datagen" / "model1.toml"


def write_datagen(tmp_path, duration=0.5, seed=3):
    """Short copy of model1.toml: one episode per reference per split."""
    path = tmp_path / "tiny.toml"
    path.write_text(f"""
[datagen]
name = "tiny"
seed = {seed}
duration = {duration}
vehicle_config = "{REPO_ROOT / 'configs' / 'quadrotor.toml'}"
controller_config = "{REPO_ROOT / 'configs' / 'baseline.toml'}"

[perturbation]
bounds = {BOUNDS.tolist()}

[wind]
type = "gaussian"
mean = {WIND_MEAN.tolist()}
cov = {WIND_COV.tolist()}
hold_period = 0.1

[references.circle]
type = "circle"
center = [0.0, 0.0, -2.0]
radius = 2.0
vertical_amplitude = 0.5
period = 7.0
yaw_deg = 0.0

[references.hover]
type = "setpoint"
position = [0.0, 0.0, -1.0]
yaw_deg = 0.0

[splits.train]
references = ["circle", "hover"]
repeats = 1

[splits.validation]
references = ["circle"]
repeats = 1
""")
    return path


def load_script():
    spec = importlib.util.spec_from_file_location(
        "generate_data", REPO_ROOT / "scripts" / "01_generate_data.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


# --- perturbations -----------------------------------------------------------

def test_perturbations_stay_in_bounds_and_are_seeded():
    u_d = random_perturbations(500, BOUNDS, np.random.default_rng(1))
    assert u_d.shape == (500, 4)
    assert np.all(np.abs(u_d) <= BOUNDS)
    # Uniform on (−a, a) has std a/√3; loose check that the full range is used.
    np.testing.assert_allclose(u_d.std(axis=0), BOUNDS / np.sqrt(3), rtol=0.15)
    np.testing.assert_array_equal(u_d, random_perturbations(500, BOUNDS, np.random.default_rng(1)))


def test_perturbations_reject_negative_bounds():
    with pytest.raises(ValueError, match="non-negative"):
        random_perturbations(10, [0.3, -0.3, 0.3, 1.0], np.random.default_rng(0))


def test_held_perturbations_index_by_period():
    values = np.arange(12.0).reshape(3, 4)
    hook = HeldPerturbations(values, 0.1)
    x, x_ref = np.zeros(13), np.zeros(10)
    np.testing.assert_array_equal(hook(0.0, x, x_ref), values[0])
    np.testing.assert_array_equal(hook(0.2, x, x_ref), values[2])  # 0.2/0.1 is not exactly 2.0


def test_held_perturbations_reject_misaligned_and_overrun():
    hook = HeldPerturbations(np.zeros((3, 4)), 0.1)
    with pytest.raises(ValueError, match="multiple"):
        hook(0.05, np.zeros(13), np.zeros(10))
    with pytest.raises(ValueError, match="past"):
        hook(0.3, np.zeros(13), np.zeros(10))


# --- Gaussian wind -----------------------------------------------------------

def test_gaussian_wind_is_held_within_a_period():
    wind = GaussianWind(WIND_MEAN, WIND_COV, 0.1, np.random.default_rng(0))
    first = wind.sample(0.0, np.zeros(3))
    np.testing.assert_array_equal(wind.sample(0.099, np.zeros(3)), first)
    assert not np.array_equal(wind.sample(0.1, np.zeros(3)), first)


def test_gaussian_wind_matches_mean_and_covariance():
    wind = GaussianWind(WIND_MEAN, WIND_COV, 0.1, np.random.default_rng(0))
    draws = np.array([wind.sample(0.1 * k, np.zeros(3)) for k in range(20000)])
    np.testing.assert_allclose(draws.mean(axis=0), WIND_MEAN, atol=0.02)
    np.testing.assert_allclose(np.cov(draws.T), WIND_COV, atol=0.02)


def test_gaussian_wind_does_not_depend_on_query_rate():
    # Querying at 1 kHz or once per hold must give the same held values.
    fast = GaussianWind(WIND_MEAN, WIND_COV, 0.1, np.random.default_rng(5))
    slow = GaussianWind(WIND_MEAN, WIND_COV, 0.1, np.random.default_rng(5))
    fast_values = [fast.sample(k * 1e-3, np.zeros(3)) for k in range(1000)][::100]
    slow_values = [slow.sample(k * 0.1, np.zeros(3)) for k in range(10)]
    np.testing.assert_array_equal(fast_values, slow_values)


def test_gaussian_wind_rejects_bad_cov_and_backwards_time():
    with pytest.raises(ValueError, match="positive definite"):
        GaussianWind(WIND_MEAN, -np.eye(3), 0.1, np.random.default_rng(0))
    wind = GaussianWind(WIND_MEAN, WIND_COV, 0.1, np.random.default_rng(0))
    wind.sample(0.5, np.zeros(3))
    with pytest.raises(ValueError, match="time order"):
        wind.sample(0.2, np.zeros(3))


# --- circle reference --------------------------------------------------------

def make_circle(yaw=0.0) -> CircleReference:
    return CircleReference([1.0, -2.0, -3.0], 2.0, 0.5, 7.0, yaw, G)


def test_circle_closed_form_points():
    reference = make_circle()
    np.testing.assert_allclose(reference(0.0)[0][POS], [1.0, -4.0, -3.0], atol=1e-12)
    np.testing.assert_allclose(reference(7.0 / 4)[0][POS], [3.0, -2.0, -3.0], atol=1e-12)


@pytest.mark.parametrize("yaw", [0.0, 0.7], ids=["north", "yawed"])
def test_circle_flat_feedforward_reproduces_reference_motion(yaw):
    reference = make_circle(yaw)
    h = 1e-5
    for t in np.linspace(0.0, 7.0, 13):
        x_ref, u_ref = reference(t)
        numerical = (reference(t + h)[0] - reference(t - h)[0]) / (2 * h)
        np.testing.assert_allclose(nominal_dynamics(x_ref, u_ref, G), numerical, atol=1e-6)


def test_circle_is_not_planar():
    reference = make_circle()
    points = np.array([reference(t)[0][POS] for t in np.linspace(0.0, 7.0, 200)])
    singular_values = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    assert singular_values[-1] > 0.1 * singular_values[0]


# --- random sum-of-sines reference --------------------------------------------

def make_random(seed=0, peak_tilt=0.3, max_speed=10.0, vertical_amplitude=0.8):
    return sample_sum_of_sines(np.random.default_rng(seed), [0.0, 0.0, -2.0], 3, [0.07, 0.25],
                               peak_tilt, max_speed, vertical_amplitude, 0.0, G)


def test_sum_of_sines_single_component_matches_closed_form():
    reference = SumOfSinesReference([1.0, 2.0, -3.0], [[2.0, 0.0, 0.5]],
                                    [[0.25, 0.1, 0.5]], [[0.0, 0.0, np.pi / 2]], 0.0, G)
    p, v, a, _ = reference.derivatives(1.0)    # ω_x t = π/2: x at its peak, at rest in x
    np.testing.assert_allclose(p, [3.0, 2.0, -3.0 + 0.5 * np.cos(np.pi)], atol=1e-12)
    assert v[0] == pytest.approx(0.0, abs=1e-12)
    assert a[0] == pytest.approx(-2.0 * (np.pi / 2) ** 2)


def test_sum_of_sines_flat_feedforward_reproduces_reference_motion():
    reference = make_random()
    h = 1e-5
    for t in np.linspace(0.0, 14.0, 15):
        x_ref, u_ref = reference(t)
        numerical = (reference(t + h)[0] - reference(t - h)[0]) / (2 * h)
        np.testing.assert_allclose(nominal_dynamics(x_ref, u_ref, G), numerical, atol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_sampled_reference_respects_its_bounds(seed):
    reference = make_random(seed, peak_tilt=0.3, max_speed=2.0, vertical_amplitude=0.8)
    t = np.linspace(0.0, 200.0, 20001)
    p, v, a = (np.array(x) for x in zip(*(reference.derivatives(ti)[:3] for ti in t)))
    tilt = np.arctan2(np.linalg.norm(a[:, :2], axis=1), G - a[:, 2])
    assert tilt.max() <= 0.3 + 1e-9
    assert np.linalg.norm(v[:, :2], axis=1).max() <= 2.0 + 1e-9
    assert np.abs(p[:, 2] + 2.0).max() <= 0.8 + 1e-9


def test_sampled_reference_reaches_the_tilt_bound_when_speed_is_free():
    """With a loose speed bound, the tilt bound is the active one: the guarantee is tight."""
    bounds = make_random(max_speed=1e3).derivative_bounds()
    assert np.arctan2(np.linalg.norm(bounds[2, :2]), G - bounds[2, 2]) == pytest.approx(0.3)


def test_sampled_reference_is_reproducible_from_the_seed():
    first, again, other = make_random(1), make_random(1), make_random(2)
    np.testing.assert_array_equal(first.amplitudes, again.amplitudes)
    assert not np.array_equal(first.amplitudes, other.amplitudes)


@pytest.mark.parametrize("kwargs, message", [
    ({"peak_tilt": 0.0}, "peak_tilt"), ({"max_speed": -1.0}, "max_speed"),
    ({"vertical_amplitude": 50.0}, "reaches g")])
def test_sampler_rejects_bad_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_random(**kwargs)


def test_outer_loop_datagen_config_flies_random_references_within_the_mpc_limits():
    """Shipped config: no circles in train/validation, every reference inside max_tilt."""
    from experiment.builders import load_toml
    from experiment.ltv_mpc_config import load_ltv_mpc
    spec = importlib.util.spec_from_file_location(
        "collect_outer_loop_data", REPO_ROOT / "scripts" / "collect_outer_loop_data.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    path = REPO_ROOT / "configs" / "datagen" / "outer_loop.toml"
    cfg = load_toml(path)
    mpc = load_ltv_mpc(path.parent / cfg["datagen"]["mpc_config"])
    assert mpc.model_drag_scale == 0.0                    # flown by the nominal controller
    rng = np.random.default_rng(0)
    for split in ("train", "validation"):
        specs = script.episode_specs(cfg["splits"][split], cfg["common"],
                                     cfg["random_reference"], G, rng)
        names = [name for name, _, _ in specs]
        assert not any(name.startswith("circle") for name in names)
        for name, ref, x0 in specs:
            if name.startswith("random"):
                bounds = ref.derivative_bounds()
                tilt = np.arctan2(np.linalg.norm(bounds[2, :2]), G - bounds[2, 2])
                assert tilt <= mpc.settings.max_tilt
                np.testing.assert_allclose(x0, state_on_reference(ref))


# --- tracking error ----------------------------------------------------------

def test_tracking_error_is_zero_on_reference():
    reference = make_circle()
    x = np.array([state_on_reference(reference, t) for t in (0.0, 1.0, 2.0)])
    x_ref = np.array([reference(t)[0] for t in (0.0, 1.0, 2.0)])
    np.testing.assert_allclose(tracking_error(x, x_ref), 0.0, atol=1e-12)


def test_tracking_error_known_offsets_and_yaw_wrap():
    yaw_x = np.radians(179.0)
    yaw_ref = np.radians(-179.0)
    x = np.zeros((1, 13))
    x[0, rigid_body.POS] = [1.0, 2.0, 3.0]
    x[0, rigid_body.VEL] = [0.5, 0.0, 0.0]
    x[0, rigid_body.QUAT] = Quaternion.from_euler([0.0, 0.0, yaw_x]).to_numpy()
    x_ref = np.zeros((1, 10))
    x_ref[0, 3:7] = Quaternion.from_euler([0.0, 0.0, yaw_ref]).to_numpy()

    e_x = tracking_error(x, x_ref)[0]
    np.testing.assert_allclose(e_x[ERR_POS], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(e_x[ERR_VEL], [0.5, 0.0, 0.0])
    # 179° − (−179°) is a 2° error the short way round, not 358°.
    np.testing.assert_allclose(e_x[ERR_RPY], [0.0, 0.0, np.radians(-2.0)], atol=1e-12)


def test_tracking_error_rejects_mismatched_rows():
    with pytest.raises(ValueError, match="x_ref"):
        tracking_error(np.zeros((3, 13)), np.zeros((2, 10)))


# --- config, script and dataset ----------------------------------------------

def test_shipped_model1_config_matches_paper_layout():
    cfg = load_datagen(MODEL1)
    splits = [e.split for e in cfg.episodes]
    assert splits.count("train") == 8 and splits.count("validation") == 2
    assert cfg.duration == 14.0
    assert sorted(cfg.references) == ["circle", "figure8"]


def test_unknown_reference_in_split_raises(tmp_path):
    path = write_datagen(tmp_path)
    path.write_text(path.read_text().replace('references = ["circle"]', 'references = ["helix"]'))
    with pytest.raises(ValueError, match="unknown reference 'helix'"):
        load_datagen(path)


def test_generate_script_writes_reproducible_dataset(tmp_path):
    script = load_script()
    config = write_datagen(tmp_path)
    run_a = script.main([str(config), "--results-dir", str(tmp_path / "a")])
    run_b = script.main([str(config), "--results-dir", str(tmp_path / "b")])
    data_a, data_b = load_dataset(run_a), load_dataset(run_b)

    assert sorted(data_a) == ["train", "validation"]
    train = data_a["train"]
    assert train.e_x.shape == (2, 5, 9)  # 2 episodes, 0.5 s at 10 Hz
    assert train.n_pairs == 2 * 4
    assert list(train.reference) == ["circle", "hover"]
    assert np.all(np.abs(train.u_d) <= BOUNDS)
    # Episodes start on their reference, so the first tracking error is zero.
    np.testing.assert_allclose(train.e_x[:, 0], 0.0, atol=1e-12)
    # Different episodes get different wind and perturbation draws.
    assert not np.array_equal(train.u_d[0], train.u_d[1])
    assert not np.array_equal(train.v_w[0], data_a["validation"].v_w[0])
    # Same config and seed -> bit-identical dataset.
    for name in ("train", "validation"):
        for field in ("e_x", "u_d", "v_w", "x"):
            np.testing.assert_array_equal(getattr(data_a[name], field),
                                          getattr(data_b[name], field))


def test_dataset_round_trips_exactly(tmp_path):
    script = load_script()
    splits = script.generate(load_datagen(write_datagen(tmp_path, duration=0.3)))
    save_dataset(tmp_path, splits)
    loaded = load_dataset(tmp_path)
    for name, split in splits.items():
        for field in ("t", "reference", "repeat", "e_x", "u_d", "v_w", "x", "x_ref", "u_h",
                      "u_cmd"):
            np.testing.assert_array_equal(getattr(loaded[name], field), getattr(split, field))
