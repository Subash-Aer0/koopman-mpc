"""K-MPC: deterministic Koopman MPC with mean wind, Eqs. (15)-(16).

    min  J = (C z_np)ᵀ Q (C z_np) + Σ (C z_k)ᵀ Q (C z_k) + ν_kᵀ R ν_k
    s.t. z_{k+1} = A z_k + B ν_k + D v̄_w,   z_0 = ψ_0
         u_lb ≤ ν_k ≤ u_ub
         h_lb ≤ z_k ≤ h_ub
Output ν_0 is added to the LQR command as the perturbation u_d.
"""


class KoopmanMPC:
    def __init__(self, model, Q, R, horizon, u_bounds, h_bounds, wind_mean):
        raise NotImplementedError

    def __call__(self, error_state):
        raise NotImplementedError
