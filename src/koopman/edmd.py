"""EDMD with l1 regularization (Bruder et al. [15]), Eqs. (7)-(11).

    min_M ||Γ2 − M Γ1||² + λ ||M||_1
with α1[k] = [ψ(x_k), u_{d,k}, v_{w,k}]ᵀ, α2[k] = [ψ(y_k), 0]ᵀ and
M = [[A, B, D], [0, 0, 0]].
"""


def build_snapshots(errors, inputs, winds, lifting):
    """Return (Γ1, Γ2) from trajectory data."""
    raise NotImplementedError


def fit_edmd(gamma1, gamma2, state_dim, input_dim, dist_dim, l1_lambda):
    """Solve the regularized least squares; return (A, B, D)."""
    raise NotImplementedError
