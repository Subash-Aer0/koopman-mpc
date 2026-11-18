"""Rotor models (Sec. II-A).

- Quadratic rotor model: thrust/torque ∝ Ω², used to compute commanded RPM
  and to estimate executed thrust f_r from measured Ω_R (Model 2).
- Hybrid Blade Element Momentum Theory (BEMT) from Davoudi et al. [1]:
  solves for inflow ratio λ0 and thrust coefficients so that momentum theory
  and blade element theory agree, giving the executed thrust and RPM under
  relative airflow.
"""


def quadratic_thrust_torque(rotor_speeds, params):
    raise NotImplementedError


def mixer(collective_thrust, body_torque, params):
    """Invert the quadratic model: (f_c, τ) -> commanded rotor speeds."""
    raise NotImplementedError


def bemt_thrust(rotor_speeds_cmd, relative_airspeed_body, params):
    """Hybrid BEMT: return executed thrust, torque and rotor speeds Ω_R."""
    raise NotImplementedError
