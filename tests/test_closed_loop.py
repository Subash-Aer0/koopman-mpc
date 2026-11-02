"""Closed-loop simulation: LQR + body-rate loop on the full nonlinear model (NED)."""
import numpy as np
import pytest

from controllers import sdlqr_model as lqr_model
from controllers.low_level import BodyRateController
from controllers.lqr import LQRTracker
from dynamics import dynamics as rigid_body
from dynamics.quaternion import Quaternion
from simulation.closed_loop import Timing, lqr_state, simulate
from wind.wind_model import ConstantWind

# Weights and limits from [16], Sec. III-B.
Q_PAPER = np.diag([100, 100, 100, 1, 1, 1, 1, 10, 10, 10.0])
R_PAPER = np.diag([1, 5, 5, 0.1])
RATE_GAINS = np.array([20.0, 20.0, 10.0])
TIMING = Timing()
IDENTITY = Quaternion.identity().to_numpy()
NO_WIND = ConstantWind(np.zeros(3))


def hover_x0() -> np.ndarray:
    x = np.zeros(rigid_body.STATE_DIM)
    x[rigid_body.QUAT] = IDENTITY
    return x


def constant_reference(params, p=np.zeros(3), q=IDENTITY):
    x_ref = np.concatenate([p, q, np.zeros(3)])
    u_ref = np.array([0.0, 0.0, 0.0, params.gravity])
    return lambda t: (x_ref, u_ref)


def run(params, duration, reference=None, wind=NO_WIND, **kwargs):
    lqr = LQRTracker(Q_PAPER, R_PAPER, TIMING.dt_control, 2.0, (2.0, 18.0))
    return simulate(
        hover_x0(), duration, reference or constant_reference(params), wind, params,
        lqr, BodyRateController(params, RATE_GAINS), TIMING, **kwargs,
    )


def test_lqr_state_matches_sdlqr_layout():
    x = np.arange(rigid_body.STATE_DIM, dtype=float)
    x_lqr = lqr_state(x)
    np.testing.assert_array_equal(x_lqr[lqr_model.POS], x[rigid_body.POS])
    np.testing.assert_array_equal(x_lqr[lqr_model.QUAT], x[rigid_body.QUAT])
    np.testing.assert_array_equal(x_lqr[lqr_model.VEL], x[rigid_body.VEL])


def test_hover_without_wind_stays_at_hover(params):
    log = run(params, 2.0)
    np.testing.assert_allclose(log.x, np.tile(hover_x0(), (len(log.t), 1)), atol=1e-10)


def test_step_reference_converges_through_full_model(params):
    # 1.7 m position step plus 30° yaw through the rate loop, drag and RK4:
    # checks the time-scale separation assumption of [16] end to end.
    q_ref = Quaternion.from_euler([0.0, 0.0, np.radians(30.0)]).to_numpy()
    reference = constant_reference(params, p=[1.0, -1.0, -1.0], q=q_ref)
    log = run(params, 20.0, reference)

    x_final = log.x[-1]
    np.testing.assert_allclose(x_final[rigid_body.POS], [1.0, -1.0, -1.0], atol=1e-2)
    np.testing.assert_allclose(x_final[rigid_body.VEL], 0.0, atol=1e-2)
    assert abs(np.dot(x_final[rigid_body.QUAT], q_ref)) == pytest.approx(1.0, abs=1e-3)


def test_constant_wind_leaves_downwind_steady_state_offset(params):
    # No integral action and no wind model in the LQR: holding the tilt that
    # balances drag needs a non-zero position error. This is the baseline
    # weakness the Koopman MPC is meant to remove.
    log = run(params, 15.0, wind=ConstantWind([3.0, 0.0, 0.0]))
    p_final = log.x[-1, rigid_body.POS]
    v_final = log.x[-1, rigid_body.VEL]

    np.testing.assert_allclose(v_final, 0.0, atol=1e-3)
    assert 0.01 < p_final[0] < 1.0
    assert abs(p_final[1]) < 1e-6


def test_log_samples_at_log_rate(params):
    log = run(params, 1.0)
    np.testing.assert_allclose(log.t, np.arange(10) * TIMING.dt_log, atol=1e-12)
    assert log.x.shape == (10, rigid_body.STATE_DIM)
    np.testing.assert_array_equal(log.u_d, 0.0)


def test_augmentation_called_at_augment_rate_and_added(params):
    calls = []
    u_d = np.array([0.0, 0.0, 0.0, 0.5])

    def augmentation(t, x, x_ref):
        calls.append(t)
        return u_d

    log = run(params, 1.0, augmentation=augmentation)
    np.testing.assert_allclose(calls, np.arange(10) * TIMING.dt_augment, atol=1e-12)
    np.testing.assert_allclose(log.u_cmd, log.u_h + u_d, atol=1e-12)


def test_augmented_command_is_saturated(params):
    log = run(params, 0.5, augmentation=lambda t, x, x_ref: np.array([5.0, 0.0, 0.0, 100.0]))
    assert np.all(np.linalg.norm(log.u_cmd[:, lqr_model.RATE], axis=1) <= 2.0 + 1e-12)
    assert np.all(log.u_cmd[:, lqr_model.THRUST] <= 18.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt_control": 0.015},
        {"dt_gain": 0.105},
        {"dt_log": 0.0},
    ],
    ids=["control_not_multiple_of_sim", "gain_not_multiple_of_control", "zero_log_period"],
)
def test_invalid_timing_raises(kwargs):
    with pytest.raises(ValueError):
        Timing(**kwargs)


def test_lqr_dt_mismatch_raises(params):
    lqr = LQRTracker(Q_PAPER, R_PAPER, 0.02, 2.0, (2.0, 18.0))
    with pytest.raises(ValueError):
        simulate(hover_x0(), 1.0, constant_reference(params), NO_WIND, params,
                 lqr, BodyRateController(params, RATE_GAINS), TIMING)
