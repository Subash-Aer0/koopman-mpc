"""Reference trajectories: closed-form positions and flat feedforward consistency."""
import numpy as np
import pytest

from controllers.sdlqr_model import POS, QUAT, RATE, THRUST, VEL, nominal_dynamics
from datagen.trajectories import Figure8Reference, flat_state

G = 9.81
CENTER = np.array([1.0, -2.0, -3.0])
AMPLITUDES = np.array([3.0, 1.5, 1.0])
PERIOD = 10.0


def make_figure8(yaw=0.0) -> Figure8Reference:
    return Figure8Reference(CENTER, AMPLITUDES, PERIOD, yaw, G)


def test_figure8_closed_form_points():
    reference = make_figure8()
    A, B, C = AMPLITUDES
    np.testing.assert_allclose(reference(0.0)[0][POS], CENTER + [0.0, 0.0, -C], atol=1e-12)
    np.testing.assert_allclose(reference(PERIOD / 4)[0][POS], CENTER + [A, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(reference(PERIOD / 2)[0][POS], CENTER + [0.0, 0.0, C], atol=1e-12)


def test_figure8_is_periodic():
    reference = make_figure8()
    for x, y in zip(reference(1.3), reference(1.3 + PERIOD)):
        np.testing.assert_allclose(x, y, atol=1e-10)


def test_figure8_is_not_planar():
    reference = make_figure8()
    points = np.array([reference(t)[0][POS] for t in np.linspace(0.0, PERIOD, 200)])
    singular_values = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    # Smallest spread direction still carries real extent: no plane fits the curve.
    assert singular_values[-1] > 0.1 * singular_values[0]


@pytest.mark.parametrize("yaw", [0.0, 0.7], ids=["north", "yawed"])
def test_flat_feedforward_reproduces_reference_motion(yaw):
    # The core flatness check: applying u_ref at x_ref must give d/dt x_ref.
    # Catches sign or frame errors in thrust, attitude and body rates.
    reference = make_figure8(yaw)
    h = 1e-5
    for t in np.linspace(0.0, PERIOD, 13):
        x_ref, u_ref = reference(t)
        x_plus, x_minus = reference(t + h)[0], reference(t - h)[0]
        numerical = (x_plus - x_minus) / (2 * h)
        np.testing.assert_allclose(nominal_dynamics(x_ref, u_ref, G), numerical, atol=1e-6)


def test_flat_state_at_rest_is_hover():
    x_ref, u_ref = flat_state(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 0.0, G)
    np.testing.assert_allclose(x_ref[QUAT], [1.0, 0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(x_ref[VEL], 0.0)
    np.testing.assert_allclose(u_ref[RATE], 0.0, atol=1e-12)
    assert u_ref[THRUST] == pytest.approx(G)


def test_flat_state_rejects_free_fall():
    with pytest.raises(ValueError, match="free fall"):
        flat_state(np.zeros(3), np.zeros(3), np.array([0.0, 0.0, G]), np.zeros(3), 0.0, G)


def test_figure8_rejects_bad_period():
    with pytest.raises(ValueError):
        Figure8Reference(CENTER, AMPLITUDES, 0.0, 0.0, G)
