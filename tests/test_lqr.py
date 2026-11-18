"""State-dependent LQR: gain properties, tracker behaviour, nominal closed loop."""
import numpy as np
import pytest

from controllers.lqr import LQRTracker, lqr_gain
from controllers.sdlqr_model import (
    POS,
    QUAT,
    RATE,
    STATE_DIM,
    THRUST,
    discretize,
    linearize,
    nominal_dynamics,
)
from dynamics.quaternion import Quaternion

G = 9.81
DT = 0.01
IDENTITY = Quaternion.identity().to_numpy()
HOVER_U = np.array([0.0, 0.0, 0.0, G])
# Weights and limits from [16], Sec. III-B.
Q_PAPER = np.diag([100, 100, 100, 1, 1, 1, 1, 10, 10, 10.0])
R_PAPER = np.diag([1, 5, 5, 0.1])
MAX_RATE = 2.0
THRUST_LIMITS = (2.0, 18.0)


def make_state(p=np.zeros(3), q=IDENTITY, v=np.zeros(3)) -> np.ndarray:
    return np.concatenate([p, q, v])


def make_tracker() -> LQRTracker:
    return LQRTracker(Q_PAPER, R_PAPER, DT, MAX_RATE, THRUST_LIMITS)


TILTED = make_state(
    p=[1.0, -2.0, -3.0],
    q=Quaternion.from_euler([0.3, -0.2, 1.1]).to_numpy(),
    v=[0.5, -1.0, 0.2],
)
MANEUVER_U = np.array([0.4, -0.7, 0.9, 12.0])
OPERATING_POINTS = pytest.mark.parametrize(
    "x, u", [(make_state(), HOVER_U), (TILTED, MANEUVER_U)], ids=["hover", "tilted"]
)


@OPERATING_POINTS
def test_gain_ignores_quaternion_norm_error(x, u):
    K = lqr_gain(x, u, Q_PAPER, R_PAPER, DT)
    assert K.shape == (4, STATE_DIM)
    np.testing.assert_allclose(K[:, QUAT] @ x[QUAT], 0.0, atol=1e-10)


@OPERATING_POINTS
def test_gain_stabilizes_all_but_the_norm_mode(x, u):
    # The full 10-state closed loop keeps exactly one eigenvalue at 1 (the
    # norm direction, untouched by K); every other mode must be strictly stable.
    K = lqr_gain(x, u, Q_PAPER, R_PAPER, DT)
    A_d, B_d = discretize(*linearize(x, u), DT)
    eigs = np.linalg.eigvals(A_d + B_d @ K)
    norm_mode = np.argmin(np.abs(eigs - 1.0))
    np.testing.assert_allclose(eigs[norm_mode], 1.0, atol=1e-9)
    assert np.max(np.abs(np.delete(eigs, norm_mode))) < 1.0


def test_hover_gain_pushes_back_toward_reference():
    # Vehicle 1 m below the reference (NED +z): thrust must rise above hover.
    tracker = make_tracker()
    tracker.update_gain(make_state(), HOVER_U)
    u = tracker(make_state(p=[0.0, 0.0, 1.0]), make_state(), HOVER_U)
    assert u[THRUST] > G


def test_zero_error_returns_feedforward():
    tracker = make_tracker()
    tracker.update_gain(TILTED, MANEUVER_U)
    np.testing.assert_allclose(tracker(TILTED, TILTED, MANEUVER_U), MANEUVER_U, atol=1e-12)


def test_reference_quaternion_sign_does_not_change_command():
    tracker = make_tracker()
    tracker.update_gain(TILTED, MANEUVER_U)
    x_ref = make_state(q=Quaternion.from_euler([0.2, -0.1, 1.0]).to_numpy())
    x_ref_flipped = x_ref.copy()
    x_ref_flipped[QUAT] *= -1
    np.testing.assert_allclose(
        tracker(TILTED, x_ref, MANEUVER_U), tracker(TILTED, x_ref_flipped, MANEUVER_U), atol=1e-12
    )


def test_command_respects_limits_and_keeps_rate_direction():
    tracker = make_tracker()
    tracker.update_gain(make_state(), HOVER_U)
    x = make_state(p=[30.0, -40.0, 50.0])
    u = tracker(x, make_state(), HOVER_U)
    u_raw = HOVER_U + tracker.K @ (x - make_state())

    assert np.linalg.norm(u[RATE]) == pytest.approx(MAX_RATE)
    assert THRUST_LIMITS[0] <= u[THRUST] <= THRUST_LIMITS[1]
    np.testing.assert_allclose(
        u[RATE] / np.linalg.norm(u[RATE]), u_raw[RATE] / np.linalg.norm(u_raw[RATE])
    )


def test_command_before_gain_raises():
    with pytest.raises(RuntimeError):
        make_tracker()(make_state(), make_state(), HOVER_U)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"Q": -Q_PAPER},
        {"R": np.zeros((4, 4))},
        {"R": np.eye(3)},
        {"dt": 0.0},
        {"max_body_rate": -1.0},
        {"thrust_limits": (18.0, 2.0)},
    ],
    ids=["negative_Q", "singular_R", "wrong_R_shape", "zero_dt", "negative_rate", "swapped_thrust"],
)
def test_invalid_construction_raises(kwargs):
    args = {
        "Q": Q_PAPER, "R": R_PAPER, "dt": DT,
        "max_body_rate": MAX_RATE, "thrust_limits": THRUST_LIMITS,
    }
    with pytest.raises(ValueError):
        LQRTracker(**{**args, **kwargs})


def _rk4_nominal(x, u, dt):
    k1 = nominal_dynamics(x, u, G)
    k2 = nominal_dynamics(x + 0.5 * dt * k1, u, G)
    k3 = nominal_dynamics(x + 0.5 * dt * k2, u, G)
    k4 = nominal_dynamics(x + dt * k3, u, G)
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    x_next[QUAT] /= np.linalg.norm(x_next[QUAT])
    return x_next


def test_nominal_closed_loop_reaches_step_reference():
    # Position step of ~1.7 m plus a 90° yaw step; gain re-solved every 10
    # control steps (10 Hz gain, 100 Hz control, as in [16], Sec. III-B).
    # 30 s because the paper's weights make yaw slow: the quaternion error is
    # ~½δθ, so Q_q = 1 against R_ωz = 5 gives a yaw time constant of ~4.5 s.
    tracker = make_tracker()
    x = make_state()
    x_ref = make_state(p=[1.0, -1.0, -1.0], q=Quaternion.from_euler([0.0, 0.0, np.pi / 2]).to_numpy())
    u = HOVER_U
    for k in range(3000):
        if k % 10 == 0:
            tracker.update_gain(x, u)
        u = tracker(x, x_ref, HOVER_U)
        x = _rk4_nominal(x, u, DT)

    np.testing.assert_allclose(x[POS], x_ref[POS], atol=1e-2)
    assert abs(np.dot(x[QUAT], x_ref[QUAT])) == pytest.approx(1.0, abs=1e-4)
