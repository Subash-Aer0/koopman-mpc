"""K-SMPC: stochastic Koopman MPC with chance constraints, Eqs. (17)-(19).

    Σζ_{k+1} = A Σζ_k Aᵀ + D Σvw Dᵀ,  Σζ_0 = 0
    γ_k = sqrt(2 diag(Σζ_k)) erf⁻¹(β)
    h_lb + γ_k ≤ z_k ≤ h_ub − γ_k
"""


def tightening(A, D, wind_cov, horizon, beta):
    """Return γ_k for k = 1..np."""
    raise NotImplementedError


class KoopmanSMPC:
    def __init__(self, model, Q, R, horizon, u_bounds, h_bounds, wind_mean, wind_cov, beta):
        raise NotImplementedError

    def __call__(self, error_state):
        raise NotImplementedError
