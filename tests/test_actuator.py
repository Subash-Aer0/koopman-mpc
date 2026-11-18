"""Actuator: ideal pass-through, BEMT equals the command in still-air hover, deviates in wind."""
import numpy as np
import pytest

from conftest import REPO_ROOT
from dynamics.actuator import RotorActuator, load_actuator
from dynamics.dynamics import QUAT, RATE, STATE_DIM, VEL
from dynamics.quaternion import Quaternion
from dynamics.rotor import load_rotor_params

HOVER_FORCE = np.array([0.0, 0.0, -9.81])
TORQUE = np.array([0.01, -0.02, 0.003])


@pytest.fixture(scope="module")
def bemt() -> RotorActuator:
    return RotorActuator("bemt", load_rotor_params(REPO_ROOT / "configs" / "rotor.toml"))


def state(velocity=(0.0, 0.0, 0.0), rate=(0.0, 0.0, 0.0), euler=(0.0, 0.0, 0.0)) -> np.ndarray:
    x = np.zeros(STATE_DIM)
    x[QUAT] = Quaternion.from_euler(euler).to_numpy()
    x[VEL] = velocity
    x[RATE] = rate
    return x


def test_ideal_passes_the_command_through():
    f, tau = RotorActuator("ideal")(state(), HOVER_FORCE, TORQUE, np.array([5.0, 0.0, 0.0]))
    np.testing.assert_array_equal(f, HOVER_FORCE)
    np.testing.assert_array_equal(tau, TORQUE)


def test_bemt_reproduces_the_command_in_still_air(bemt):
    """Calibrated BEMT = quadratic model with no airflow, so the mixer's wrench is met."""
    f, tau = bemt(state(), HOVER_FORCE, TORQUE, np.zeros(3))
    np.testing.assert_allclose(f, HOVER_FORCE, rtol=1e-9)
    np.testing.assert_allclose(tau, TORQUE, atol=1e-9)


def test_moving_with_the_wind_is_still_air(bemt):
    """Only air-relative velocity matters: v = v_w gives the commanded wrench."""
    wind = np.array([3.0, -1.0, 0.2])
    f, _ = bemt(state(velocity=wind), HOVER_FORCE, np.zeros(3), wind)
    np.testing.assert_allclose(f, HOVER_FORCE, rtol=1e-9)


def test_wind_across_the_disk_adds_thrust_and_climb_removes_it(bemt):
    f_wind, _ = bemt(state(), HOVER_FORCE, np.zeros(3), np.array([3.0, 0.0, 0.0]))
    f_climb, _ = bemt(state(velocity=(0.0, 0.0, -1.0)), HOVER_FORCE, np.zeros(3), np.zeros(3))
    assert -f_wind[2] > 9.81 * 1.05
    assert -f_climb[2] < 9.81 * 0.95


def test_yaw_rate_unbalances_nothing_but_roll_rate_does(bemt):
    """ω × r_i: a roll rate moves the left and right rotors axially in opposite directions."""
    _, tau_roll = bemt(state(rate=(2.0, 0.0, 0.0)), HOVER_FORCE, np.zeros(3), np.zeros(3))
    assert tau_roll[0] < -1e-4                      # damping: opposes the roll rate
    _, tau_yaw = bemt(state(rate=(0.0, 0.0, 2.0)), HOVER_FORCE, np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(tau_yaw[:2], 0.0, atol=1e-12)


def test_rejects_lateral_thrust_and_unknown_models(bemt):
    with pytest.raises(ValueError, match="−z_body"):
        bemt(state(), np.array([0.1, 0.0, -9.81]), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="actuator model"):
        RotorActuator("propeller")
    with pytest.raises(ValueError, match="needs rotor"):
        RotorActuator("bemt")


def test_load_actuator_reads_the_vehicle_rotor_table(tmp_path):
    vehicle = REPO_ROOT / "configs" / "quadrotor.toml"
    assert load_actuator({}, vehicle).model == "ideal"
    assert load_actuator({"rotor": {"model": "ideal"}}, vehicle).model == "ideal"
    actuator = load_actuator({"rotor": {"model": "bemt", "rotor_config": "rotor.toml"}}, vehicle)
    assert actuator.model == "bemt"
    assert actuator.config_paths == ((REPO_ROOT / "configs" / "rotor.toml").resolve(),)
    with pytest.raises(ValueError, match="rotor_config"):
        load_actuator({"rotor": {"model": "bemt"}}, vehicle)


class _ConstantCommand:
    """Outer loop that always commands level hover thrust, so only the plant differs."""
    period = 0.1

    def __init__(self, c: float):
        self.u = np.array([0.0, 0.0, 0.0, c])

    def __call__(self, t, x):
        return self.u


def _fly(actuator, wind_velocity, seconds=1.0):
    from controllers.low_level import BodyRateController
    from datagen.trajectories import SetpointReference
    from dynamics.params import QuadParams
    from simulation.outer_loop import simulate_outer_loop
    from wind.wind_model import ConstantWind

    vehicle = QuadParams(mass=1.0, inertia=np.diag([0.0082, 0.0082, 0.0149]),
                         drag_coeffs=np.zeros(3), gravity=9.81)
    return simulate_outer_loop(state(), seconds, SetpointReference(np.zeros(3), 0.0, 9.81),
                               ConstantWind(wind_velocity), vehicle, _ConstantCommand(9.81),
                               BodyRateController(vehicle, [20.0, 20.0, 10.0]),
                               actuator=actuator)


def test_bemt_vehicle_flies_like_the_ideal_one_in_still_air(bemt):
    ideal, real = _fly(None, np.zeros(3)), _fly(bemt, np.zeros(3))
    np.testing.assert_allclose(real.x, ideal.x, atol=1e-9)
    np.testing.assert_allclose(real.x[-1, :3], 0.0, atol=1e-9)        # hover holds exactly


def test_bemt_vehicle_climbs_in_a_crosswind_the_ideal_one_does_not(bemt):
    """Edgewise flow adds thrust; with no drag the ideal vehicle feels nothing."""
    wind = np.array([3.0, 0.0, 0.0])
    ideal, real = _fly(None, wind), _fly(bemt, wind)
    np.testing.assert_allclose(ideal.x[-1, :3], 0.0, atol=1e-9)
    assert real.x[-1, 2] < -0.1          # NED: risen by more than 0.1 m in 1 s


def test_experiment_carries_the_vehicle_actuator(tmp_path):
    """load_experiment builds the actuator from the vehicle's [rotor] table and records its file."""
    from experiment.builders import load_experiment
    configs = REPO_ROOT / "configs"
    vehicle = (configs / "quadrotor.toml").read_text()
    vehicle = vehicle.replace('model = "ideal"', 'model = "bemt"').replace(
        'rotor_config = "rotor.toml"', f'rotor_config = "{(configs / "rotor.toml").as_posix()}"')
    (tmp_path / "vehicle.toml").write_text(vehicle)
    experiment = (configs / "experiments" / "hover_gaussian.toml").read_text()
    experiment = experiment.replace('vehicle_config = "../quadrotor.toml"',
                                    f'vehicle_config = "{(tmp_path / "vehicle.toml").as_posix()}"')
    experiment = experiment.replace('controller_config = "../baseline.toml"',
                                    f'controller_config = "{(configs / "baseline.toml").as_posix()}"')
    (tmp_path / "exp.toml").write_text(experiment)
    exp = load_experiment(tmp_path / "exp.toml")
    assert exp.actuator.model == "bemt"
    assert (configs / "rotor.toml").resolve() in exp.config_paths
