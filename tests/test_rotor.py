"""Rotor models: quadratic model and mixer, calibrated BEMT against analytic limits."""
import numpy as np
import pytest

from conftest import REPO_ROOT
from dynamics.rotor import (
    RotorParams, bemt_inflow, bemt_thrust, body_wrench, load_rotor_params, mixer,
    quadratic_thrust_torque, rotor_airspeeds,
)


@pytest.fixture(scope="module")
def rotor() -> RotorParams:
    return load_rotor_params(REPO_ROOT / "configs" / "rotor.toml")


def still(n=4):
    return np.zeros((n, 3))


def test_mixer_inverts_the_quadratic_model(rotor):
    tau = np.array([0.02, -0.03, 0.004])
    omega = mixer(10.5, tau, rotor)
    T, Q = quadratic_thrust_torque(omega, rotor)
    f_t, tau_t = body_wrench(T, Q, rotor)
    np.testing.assert_allclose(f_t, [0.0, 0.0, -10.5], rtol=1e-12)
    np.testing.assert_allclose(tau_t, tau, atol=1e-12)


def test_hover_needs_equal_rotor_speeds(rotor):
    omega = mixer(9.81, np.zeros(3), rotor)
    np.testing.assert_allclose(omega, omega[0])
    assert rotor.thrust_coefficient * omega[0]**2 == pytest.approx(9.81 / 4)


def test_positive_roll_torque_speeds_up_the_left_rotors(rotor):
    """τ_x > 0 (right wing down in FRD) needs more thrust on the left (y < 0) rotors."""
    omega = mixer(9.81, [0.05, 0.0, 0.0], rotor)
    left = rotor.positions[:, 1] < 0
    assert np.all(omega[left] > omega[~left].max())


@pytest.mark.parametrize("omega", [300.0, 576.0, 900.0])
def test_bemt_equals_the_quadratic_model_in_still_air_hover(rotor, omega):
    """The calibration holds at every rotor speed, not only at hover thrust."""
    speeds = np.full(4, omega)
    T, Q, omega_r = bemt_thrust(speeds, still(), rotor)
    T_q, Q_q = quadratic_thrust_torque(speeds, rotor)
    np.testing.assert_allclose(T, T_q, rtol=1e-9)
    np.testing.assert_allclose(Q, Q_q, rtol=1e-9)
    np.testing.assert_array_equal(omega_r, speeds)


def test_hover_inflow_is_the_momentum_theory_value(rotor):
    lam, c_t = bemt_inflow(np.zeros(1), np.zeros(1), rotor)
    assert lam[0] == pytest.approx(np.sqrt(c_t[0] / 2.0), rel=1e-10)


@pytest.mark.parametrize("mu, lam_c", [(0.05, 0.0), (0.0, 0.02), (0.08, -0.01), (0.12, 0.03)])
def test_solved_inflow_satisfies_both_theories(rotor, mu, lam_c):
    lam, c_t = bemt_inflow(np.array([mu]), np.array([lam_c]), rotor)
    blade = (rotor.solidity * rotor.lift_slope / 2.0) * (
        rotor.root_pitch / 3.0 * (1 + 1.5 * mu**2) + rotor.twist / 4.0 * (1 + mu**2) - lam / 2.0)
    momentum = 2.0 * (lam - lam_c) * np.sqrt(mu**2 + lam**2)
    np.testing.assert_allclose(c_t, blade, rtol=1e-10)
    np.testing.assert_allclose(c_t, momentum, rtol=1e-9)


def test_climb_loses_thrust_and_edgewise_flow_gains_it(rotor):
    """Axial inflow lowers the blade angle of attack; edgewise flow adds translational lift."""
    omega = np.full(4, 576.0)
    thrust = lambda v: bemt_thrust(omega, np.tile(v, (4, 1)), rotor)[0][0]
    hover = thrust([0.0, 0.0, 0.0])
    climbs = [thrust([0.0, 0.0, -vz]) for vz in (0.5, 1.0, 2.0)]   # FRD: climb is −z
    edges = [thrust([vx, 0.0, 0.0]) for vx in (1.0, 3.0, 6.0)]
    assert hover > climbs[0] > climbs[1] > climbs[2]
    assert hover < edges[0] < edges[1] < edges[2]


def test_rotor_airspeed_adds_the_rotation_of_the_arm(rotor):
    v = rotor_airspeeds(np.zeros(3), [0.0, 0.0, 2.0], rotor)          # yaw rate only
    np.testing.assert_allclose(v, np.cross([0.0, 0.0, 2.0], rotor.positions))
    np.testing.assert_allclose(v[:, 2], 0.0)


def test_vortex_ring_state_is_refused(rotor):
    omega = np.full(4, 576.0)
    descent = 0.6 * rotor.hover_inflow * 576.0 * rotor.radius       # 0.6 v_h, past onset
    with pytest.raises(ValueError, match="vortex ring"):
        bemt_thrust(omega, np.tile([0.0, 0.0, descent], (4, 1)), rotor)


def test_rejects_a_torque_coefficient_below_induced_power(rotor):
    from dataclasses import replace
    with pytest.raises(ValueError, match="C_d0"):
        replace(rotor, torque_coefficient=rotor.torque_coefficient * 0.3)


@pytest.mark.parametrize("call", [
    lambda r: bemt_thrust(np.full(4, 500.0), np.zeros((3, 3)), r),
    lambda r: bemt_thrust(np.array([500.0, 500.0, -1.0, 500.0]), np.zeros((4, 3)), r),
    lambda r: mixer(np.nan, np.zeros(3), r),
    lambda r: quadratic_thrust_torque([-1.0, 0.0, 0.0, 0.0], r),
])
def test_rejects_bad_inputs(rotor, call):
    with pytest.raises(ValueError):
        call(rotor)


def test_a_stopped_rotor_gives_no_thrust(rotor):
    T, Q, _ = bemt_thrust(np.array([576.0, 576.0, 0.0, 576.0]), np.zeros((4, 3)), rotor)
    assert T[2] == 0.0 and Q[2] == 0.0
    assert T[0] == pytest.approx(rotor.thrust_coefficient * 576.0**2)
