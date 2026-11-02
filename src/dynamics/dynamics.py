"""13-state quadcopter model with wind, Eqs. (1)-(4).

State x = [p (3), q (4), v_r (3), ω (3)]:
    ṗ   = v_r + v_w
    q̇   = ½ q ⊗ [0, ω]
    v̇_r = g + (1/m) q ⊙ f_t − (1/m) f_D
    ω̇   = J⁻¹ τ_t − J⁻¹ [ω × (J ω)]
"""


def state_derivative(x, rotor_speeds, wind_velocity, params):
    """Continuous-time dynamics ẋ = f(x, Ω, v_w)."""
    raise NotImplementedError


def step(x, rotor_speeds, wind_velocity, params, dt):
    """Integrate the dynamics over one time step (e.g. RK4)."""
    raise NotImplementedError
