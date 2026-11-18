"""LTV MPC: reduces to the hover MPC at a setpoint, to TV-LQR unconstrained, and tracks a fast circle."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from controllers.hover_model import EULER, HoverModel, POS, VEL, euler_dynamics
from controllers.linear_mpc import LinearizedReference, LinearMPC, LinearMPCSettings
from controllers.ltv_mpc import LTVMPC
from controllers.trajectory_model import LTVModel
from datagen.trajectories import CircleReference
from dynamics import dynamics as rigid_body
from dynamics.params import QuadParams
from dynamics.quaternion import Quaternion

WIND = np.array([2.7032, 1.7566, 0.0124])
DT = 0.1
HORIZON = 20
LEAD = 5.0   # s, Riccati lead; one lap of the 5 s circle


@pytest.fixture
def vehicle() -> QuadParams:
    return QuadParams(mass=1.0, inertia=np.diag([0.0082, 0.0082, 0.0149]),
                      drag_coeffs=np.array([0.05, 0.05, 0.10]), gravity=9.81)


def settings(terminal="cost_only", loose=False) -> LinearMPCSettings:
    """The weights and limits of configs/controllers/hover_mpc.toml; loose = no active bound."""
    rate = 1e3 if loose else 2.0 / np.sqrt(3.0)
    thrust = (-1e3, 1e3) if loose else (2.0, 18.0)
    return LinearMPCSettings(
        Q=np.diag([100.0] * 3 + [10.0] * 3 + [1.0] * 3), R=np.diag([1.0, 5.0, 5.0, 0.1]),
        horizon=HORIZON, terminal=terminal, u_lb=np.array([-rate] * 3 + [thrust[0]]),
        u_ub=np.array([rate] * 3 + [thrust[1]]), max_tilt=1.5 if loose else 0.5)


def setpoint(position):
    position = np.asarray(position, dtype=float)
    return lambda t: (position, np.zeros(3), np.zeros(3), np.zeros(3))


def full_state(x9) -> np.ndarray:
    """9-state [p, v, η] -> 13-state with zero body rates."""
    x = np.zeros(rigid_body.STATE_DIM)
    x[rigid_body.POS] = x9[POS]
    x[rigid_body.VEL] = x9[VEL]
    x[rigid_body.QUAT] = Quaternion.from_euler(x9[EULER]).to_numpy()
    return x


def ltv_mpc(vehicle, derivatives, run, terminal="cost_only", loose=False, yaw=0.0):
    model = LTVModel.build(derivatives, run + HORIZON * DT + LEAD, vehicle, WIND, yaw, DT)
    return LTVMPC(model, settings(terminal, loose), LEAD)


@pytest.mark.parametrize("terminal", ["cost_only", "equality"])
def test_at_a_setpoint_ltv_mpc_is_the_hover_mpc(vehicle, terminal):
    target = setpoint([1.0, -2.0, -3.0])
    ltv = ltv_mpc(vehicle, target, 2.0, terminal)
    hover_model = HoverModel.build(vehicle, WIND, yaw=0.0, dt=DT)
    hover = LinearMPC(hover_model, settings(terminal), LinearizedReference(hover_model, target), 2.0)
    rng = np.random.default_rng(1)
    for k in range(5):
        x9 = hover_model.x_trim + np.r_[1.0, -2.0, -3.0, np.zeros(6)]
        x9 = x9 + 0.1 * rng.standard_normal(9)
        np.testing.assert_allclose(ltv(k * DT, full_state(x9)), hover(k * DT, full_state(x9)),
                                   atol=1e-6)


def test_unconstrained_without_residual_is_time_varying_lqr(vehicle):
    circle = CircleReference([0.0, 0.0, -2.0], 2.0, 0.5, 5.0, 0.0, 9.81)
    mpc = ltv_mpc(vehicle, circle.derivatives, 3.0, loose=True)
    # Zero the residual so the optimum has no feed-forward part: Δ_0 = K_k e_0 exactly.
    mpc = LTVMPC(dataclasses.replace(mpc.model, r=np.zeros_like(mpc.model.r)), mpc.settings, LEAD)
    rng = np.random.default_rng(2)
    for k in (0, 11, 23):
        x9 = mpc.model.x_ref[k] + 0.05 * rng.standard_normal(9)
        e0 = mpc.error_state(k * DT, full_state(x9))
        u = mpc(k * DT, full_state(x9))
        np.testing.assert_allclose(u - mpc.model.u_ref[k], mpc.K[k] @ e0, atol=1e-8)


def _closed_loop_error(controller, circle, vehicle, laps=1):
    """Max position error flying `circle` on the 9-state nonlinear model (ideal rate loop)."""
    from controllers.trajectory_model import trajectory_point
    x9, _ = trajectory_point(*circle.derivatives(0.0), circle.yaw, WIND, vehicle)
    worst = 0.0
    for k in range(int(round(laps * 2 * np.pi / circle.omega / DT))):
        u = controller(k * DT, full_state(x9))
        x9 = solve_ivp(lambda _, s: euler_dynamics(s, u, WIND, vehicle), (0.0, DT), x9,
                       rtol=1e-10, atol=1e-10).y[:, -1]
        worst = max(worst, np.linalg.norm(x9[POS] - circle.derivatives((k + 1) * DT)[0]))
    return worst


def test_ltv_tracks_the_fast_circle_that_the_hover_model_cannot(vehicle):
    # Same plant the models were derived from (no rate lag, no turbulence), so the
    # only difference between the two controllers is the linearization point.
    circle = CircleReference([0.0, 0.0, -2.0], 2.0, 0.0, 5.0, 0.0, 9.81)
    run = 5.0
    ltv = ltv_mpc(vehicle, circle.derivatives, run)
    hover_model = HoverModel.build(vehicle, WIND, yaw=0.0, dt=DT)
    hover = LinearMPC(hover_model, settings("cost_only"),
                      LinearizedReference(hover_model, circle.derivatives), run)
    ltv_error = _closed_loop_error(ltv, circle, vehicle)
    hover_error = _closed_loop_error(hover, circle, vehicle)
    assert ltv_error < 1e-3
    assert hover_error > 20 * ltv_error
    assert ltv.fallback_count == 0


def test_calls_off_the_grid_or_past_the_usable_range_are_rejected(vehicle):
    mpc = ltv_mpc(vehicle, setpoint([0.0, 0.0, -1.0]), 2.0)
    x = full_state(mpc.model.x_ref[0])
    with pytest.raises(ValueError, match="sample grid"):
        mpc(0.05, x)
    with pytest.raises(ValueError, match="usable range"):
        mpc(mpc.last_index * DT + DT, x)
    assert mpc.last_index * DT == pytest.approx(2.0)


def test_unsupported_or_inconsistent_setups_are_rejected(vehicle):
    target = setpoint([0.0, 0.0, -1.0])
    model = LTVModel.build(target, 1.0, vehicle, WIND, 0.0, DT)
    with pytest.raises(ValueError, match="terminal"):
        LTVMPC(model, settings("polytope"), 0.0)
    with pytest.raises(ValueError, match="too short"):
        LTVMPC(model, settings(), LEAD)
    fast = CircleReference([0.0, 0.0, -2.0], 4.0, 0.0, 3.0, 0.0, 9.81)   # needs ~1.06 rad tilt
    long_model = LTVModel.build(fast.derivatives, 5.0, vehicle, WIND, 0.0, DT)
    with pytest.raises(ValueError, match="tilt"):
        LTVMPC(long_model, settings(), 0.0)


REPO_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "controllers"


def test_ltv_config_shares_everything_but_the_terminal_with_its_base():
    from experiment.hover_mpc_config import load_hover_mpc
    from experiment.ltv_mpc_config import load_ltv_mpc
    cfg = load_ltv_mpc(REPO_CONFIGS / "ltv_mpc.toml")
    base = load_hover_mpc(REPO_CONFIGS / "hover_mpc.toml")
    assert cfg.base.config_path == base.config_path
    assert cfg.settings.terminal == cfg.terminal
    for name in ("Q", "R", "u_lb", "u_ub"):
        np.testing.assert_array_equal(getattr(cfg.settings, name), getattr(base.settings, name))
    assert (cfg.settings.horizon, cfg.settings.max_tilt) == (base.settings.horizon,
                                                             base.settings.max_tilt)


def test_ltv_config_model_covers_the_run(vehicle):
    from experiment.ltv_mpc_config import load_ltv_mpc
    cfg = load_ltv_mpc(REPO_CONFIGS / "ltv_mpc.toml")
    mpc = cfg.make_controller(cfg.make_model(vehicle, setpoint([0.0, 0.0, -1.0]), 2.0))
    assert mpc.last_index * mpc.period == pytest.approx(2.0)


@pytest.mark.parametrize("line, message", [('terminal = "polytope"', "terminal"),
                                           ("riccati_lead = -1.0", "riccati_lead"),
                                           ("model_drag_scale = -0.5", "model_drag_scale"),
                                           ("model_drag_scale = nan", "model_drag_scale")])
def test_ltv_config_rejects_bad_values(tmp_path, line, message):
    from experiment.ltv_mpc_config import load_ltv_mpc
    key = line.split(" =")[0]
    values = {"base_config": f'"{(REPO_CONFIGS / "hover_mpc.toml").as_posix()}"',
              "terminal": '"cost_only"', "riccati_lead": "5.0", "model_drag_scale": "1.0"}
    body = "\n".join(f"{k} = {v}" for k, v in values.items() if k != key)
    path = tmp_path / "ltv.toml"
    path.write_text(f"[ltv_mpc]\n{body}\n{line}\n")
    with pytest.raises(ValueError, match=message):
        load_ltv_mpc(path)


def test_nominal_and_oracle_configs_differ_only_in_the_believed_drag(vehicle):
    from experiment.ltv_mpc_config import load_ltv_mpc
    oracle = load_ltv_mpc(REPO_CONFIGS / "ltv_mpc.toml")
    nominal = load_ltv_mpc(REPO_CONFIGS / "ltv_mpc_nominal.toml")
    assert nominal.base.config_path == oracle.base.config_path
    assert (nominal.terminal, nominal.riccati_lead) == (oracle.terminal, oracle.riccati_lead)
    assert (oracle.model_drag_scale, nominal.model_drag_scale) == (1.0, 0.0)

    believed = nominal.model_params(vehicle)
    np.testing.assert_array_equal(believed.drag_coeffs, np.zeros(3))
    assert (believed.mass, believed.gravity) == (vehicle.mass, vehicle.gravity)
    np.testing.assert_array_equal(believed.inertia, vehicle.inertia)
    np.testing.assert_array_equal(oracle.model_params(vehicle).drag_coeffs, vehicle.drag_coeffs)


def test_nominal_hover_trim_ignores_the_wind(vehicle):
    """Without drag the wind exerts no force, so the trim is level hover at c = g."""
    from experiment.ltv_mpc_config import load_ltv_mpc
    nominal = load_ltv_mpc(REPO_CONFIGS / "ltv_mpc_nominal.toml")
    hover = nominal.hover_model(vehicle)
    np.testing.assert_allclose(hover.x_trim[EULER][:2], 0.0, atol=1e-12)
    np.testing.assert_allclose(hover.u_trim, [0.0, 0.0, 0.0, vehicle.gravity], atol=1e-12)
    oracle = load_ltv_mpc(REPO_CONFIGS / "ltv_mpc.toml").hover_model(vehicle)
    assert np.max(np.abs(oracle.x_trim[EULER][:2])) > 1e-3   # the oracle leans into the wind


def test_greybox_belief_replaces_the_scaled_drag(vehicle):
    from experiment.ltv_mpc_config import load_ltv_mpc
    nominal = load_ltv_mpc(REPO_CONFIGS / "ltv_mpc_nominal.toml")
    greybox = nominal.with_drag_coeffs([0.04, 0.05, 0.11])
    np.testing.assert_array_equal(greybox.model_params(vehicle).drag_coeffs, [0.04, 0.05, 0.11])
    np.testing.assert_array_equal(nominal.model_params(vehicle).drag_coeffs, np.zeros(3))
    with pytest.raises(ValueError, match="non-negative"):
        nominal.with_drag_coeffs([0.04, -0.01, 0.1])
