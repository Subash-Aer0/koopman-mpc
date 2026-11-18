"""Grey-box drag fit: recovers known coefficients from simulated motion, rejects bad input."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from controllers.hover_model import euler_dynamics
from dynamics.params import QuadParams
from identification.greybox_drag import drag_regression, fit_drag, residual_rms

DT = 0.1
TRUE_DRAG = np.array([0.04, 0.06, 0.12])


def simulate_episode(seed: int, params: QuadParams, T: int = 140):
    """9-state model under random held commands and wind; dense integration per sample."""
    rng = np.random.default_rng(seed)
    x = np.zeros(9)
    x[3:6] = rng.uniform(-2.0, 2.0, 3)
    xs, us, ws = [x.copy()], [], []
    for _ in range(T - 1):
        u = np.array([*rng.uniform(-0.4, 0.4, 3), params.gravity + rng.uniform(-2.0, 2.0)])
        u[:2] -= 0.8 * x[6:8]                       # keep roll/pitch near level
        w = np.array([2.7, 1.8, 0.0]) + rng.normal(0.0, 0.7, 3)
        x = solve_ivp(lambda t, s: euler_dynamics(s, u, w, params), (0.0, DT), x,
                      rtol=1e-10, atol=1e-12).y[:, -1]
        xs.append(x.copy()); us.append(u); ws.append(w)
    us.append(us[-1]); ws.append(ws[-1])
    return np.array(xs), np.array(us), np.array(ws)


@pytest.fixture(scope="module")
def data():
    params = QuadParams(mass=1.0, inertia=np.diag([0.0082, 0.0082, 0.0149]),
                        drag_coeffs=TRUE_DRAG, gravity=9.81)
    parts = [drag_regression(*simulate_episode(seed, params), DT, params.mass, params.gravity)
             for seed in range(4)]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def test_anisotropic_fit_recovers_the_true_coefficients(data):
    d = fit_drag(*data, "anisotropic")
    np.testing.assert_allclose(d, TRUE_DRAG, rtol=0.03)
    assert np.all(residual_rms(*data, d) < 0.05)        # midpoint discretization only


def test_restricted_forms_tie_coefficients_and_fit_worse(data):
    d_iso = fit_drag(*data, "isotropic")
    d_hor = fit_drag(*data, "horizontal_equal")
    assert d_iso[0] == d_iso[1] == d_iso[2]
    assert d_hor[0] == d_hor[1]
    full = residual_rms(*data, fit_drag(*data, "anisotropic")).sum()
    assert residual_rms(*data, d_iso).sum() > 1.5 * full


def test_zero_drag_data_gives_zero_not_negative_coefficients():
    params = QuadParams(mass=1.0, inertia=np.diag([0.0082, 0.0082, 0.0149]),
                        drag_coeffs=np.zeros(3), gravity=9.81)
    X, y = drag_regression(*simulate_episode(7, params), DT, params.mass, params.gravity)
    d = fit_drag(X, y)
    assert np.all(d >= 0.0)
    np.testing.assert_allclose(d, 0.0, atol=2e-3)


@pytest.mark.parametrize("kwargs, message", [
    ({"form": "diagonal"}, "form"),
])
def test_fit_rejects_unknown_form(data, kwargs, message):
    with pytest.raises(ValueError, match=message):
        fit_drag(*data, **kwargs)


def test_regression_rejects_bad_shapes_and_nan():
    x9, u, w = np.zeros((5, 9)), np.zeros((5, 4)), np.zeros((5, 3))
    with pytest.raises(ValueError, match="need x9"):
        drag_regression(x9, u[:4], w, DT, 1.0, 9.81)
    x9[2, 4] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        drag_regression(x9, u, w, DT, 1.0, 9.81)
