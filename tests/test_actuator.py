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


# ---------------------------------------------------------------- motor model
from dynamics.motor import MotorParams  # noqa: E402
from dynamics.rotor import mixer  # noqa: E402

MOTOR = MotorParams(time_constant=0.03, rotor_inertia=3e-5)
DT = 1e-3


def motor_actuator() -> RotorActuator:
    act = RotorActuator("bemt", load_rotor_params(REPO_ROOT / "configs" / "rotor.toml"),
                        motor=MOTOR)
    act.reset(DT)
    return act


def run(act, x, v_w, steps):
    for _ in range(steps):
        f, tau = act(x, HOVER_FORCE, np.zeros(3), v_w)
    return f, tau


def test_motor_in_still_air_hover_holds_the_commanded_speed():
    act = motor_actuator()
    f, _ = run(act, state(), np.zeros(3), 200)
    omega_cmd = mixer(9.81, np.zeros(3), act.rotor)
    np.testing.assert_allclose(act.rotor_speeds, omega_cmd, rtol=1e-12)
    np.testing.assert_allclose(f, HOVER_FORCE, rtol=1e-9)


def test_edgewise_wind_slows_the_rotors_but_adds_thrust():
    """The sign problem of Model 2 under this BEMT: edgewise flow raises the torque, so the
    measured speed drops and f_res = f_c − Σ k_T Ω_R² says "less thrust", while BEMT
    actually delivers more thrust than commanded."""
    act = motor_actuator()
    f, _ = run(act, state(), np.array([5.0, 0.0, 0.0]), 300)       # 10 τ_m: settled
    omega_cmd = mixer(9.81, np.zeros(3), act.rotor)
    assert np.all(act.rotor_speeds < omega_cmd)
    f_res = 9.81 - act.rotor.thrust_coefficient * np.sum(act.rotor_speeds**2)   # m = 1 kg
    assert f_res > 0                       # residual says: thrust missing
    assert -f[2] > 9.81                    # truth: more thrust than commanded


def test_motor_needs_reset_and_reset_restarts_the_state():
    act = RotorActuator("bemt", load_rotor_params(REPO_ROOT / "configs" / "rotor.toml"),
                        motor=MOTOR)
    with pytest.raises(RuntimeError):
        act(state(), HOVER_FORCE, np.zeros(3), np.zeros(3))
    act.reset(DT)
    run(act, state(), np.array([5.0, 0.0, 0.0]), 100)
    act.reset(DT)
    assert np.all(np.isnan(act.rotor_speeds))
    act(state(), HOVER_FORCE, np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(act.rotor_speeds, mixer(9.81, np.zeros(3), act.rotor))


def test_rotor_speeds_without_motor_are_the_command_and_nan_for_ideal(bemt):
    bemt(state(), HOVER_FORCE, TORQUE, np.array([2.0, 1.0, 0.0]))
    np.testing.assert_allclose(bemt.rotor_speeds, mixer(9.81, TORQUE, bemt.rotor))
    ideal = RotorActuator("ideal")
    ideal(state(), HOVER_FORCE, TORQUE, np.zeros(3))
    assert np.all(np.isnan(ideal.rotor_speeds))
    with pytest.raises(ValueError):
        RotorActuator("ideal", motor=MOTOR)


def test_motor_vehicle_config_is_the_default_vehicle_plus_a_motor():
    import tomllib
    configs = REPO_ROOT / "configs"
    default = tomllib.loads((configs / "quadrotor.toml").read_text())
    with_motor = tomllib.loads((configs / "quadrotor_motor.toml").read_text())
    motor = with_motor.pop("motor")
    assert with_motor == default
    act = load_actuator(tomllib.loads((configs / "quadrotor_motor.toml").read_text()),
                        configs / "quadrotor_motor.toml")
    assert act.motor == MotorParams(motor["time_constant"], motor["rotor_inertia"])
    assert load_actuator(default, configs / "quadrotor.toml").motor is None
