"""EDMD with l1 regularization (Bruder et al. [15]), Eqs. (7)-(11).

    min_M ||Γ2 − M Γ1||² + λ ||M||_1
with α1[k] = [ψ(x_k), u_{d,k}, v_{w,k}]ᵀ, α2[k] = [ψ(y_k), 0]ᵀ and
M = [[A, B, D], [0, 0, 0]].

The bottom rows of Γ2 are zero and the bottom rows of M are fixed at zero, so
only the top N rows are fitted: Y = Γ2[:N] ≈ [A B D] Γ1. With λ = 0 this is
ordinary least squares; with λ > 0 it is a Lasso, solved here by coordinate descent.

Constant feature: if ψ_j ≡ 1, its true row is known, A[j] = e_jᵀ, B[j] = D[j] = 0.
The l1 penalty would shrink A[j, j] below 1, so the "1" decays over a
prediction horizon and biases every term it multiplies. The objective is a
sum of independent per-row problems, so fixing row j and fitting the others
gives exactly the other rows of the unconstrained fit.
"""
import numpy as np

_CD_MAX_SWEEPS = 100000
_CD_TOL = 1e-12  # stop when no coefficient moves more than this, relative to max |M|


def build_snapshots(errors, inputs, winds, lifting):
    """Return (Γ1, Γ2) from trajectory data.

    Pairs (k, k+1) are formed inside each episode and never across two.

    Args:
        errors: Per-episode error states, list of arrays (T_i, n).
        inputs: Per-episode u_d, list of arrays (T_i, m); sample k held over [k, k+1).
        winds: Per-episode v_w, list of arrays (T_i, d); sample k held over [k, k+1).
        lifting: ψ, maps (K, n) -> (K, N).

    Returns:
        Γ1 with columns α1[k] = [ψ(x_k); u_d,k; v_w,k] and Γ2 with columns
        α2[k] = [ψ(x_k+1); 0; 0], both shape (N + m + d, K), K = Σ (T_i − 1).
    """
    if not (len(errors) == len(inputs) == len(winds)) or not errors:
        raise ValueError("errors, inputs and winds must be equally long, non-empty lists")
    gamma1, gamma2 = [], []
    for i, (x, u, w) in enumerate(zip(errors, inputs, winds)):
        x, u, w = (np.asarray(a, dtype=float) for a in (x, u, w))
        if x.ndim != 2 or u.ndim != 2 or w.ndim != 2 or not len(x) == len(u) == len(w):
            raise ValueError(f"episode {i}: errors, inputs, winds must be 2-D with equal "
                             f"lengths, got {x.shape}, {u.shape}, {w.shape}")
        if len(x) < 2:
            raise ValueError(f"episode {i} has {len(x)} samples; a pair needs at least 2")
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(w))):
            raise ValueError(f"episode {i}: inputs or winds contain NaN or inf")
        psi = lifting(x)  # lift once, then shift: ψ(x_k) and ψ(x_k+1) share rows
        gamma1.append(np.hstack([psi[:-1], u[:-1], w[:-1]]).T)
        gamma2.append(np.hstack([psi[1:], np.zeros((len(x) - 1, u.shape[1] + w.shape[1]))]).T)
    return np.hstack(gamma1), np.hstack(gamma2)


def _lasso_coordinate_descent(gamma1: np.ndarray, Y: np.ndarray,
                              l1_lambda: float) -> np.ndarray:
    """argmin_M ||Y − M Γ1||_F² + λ ||M||_1 by cyclic coordinate descent.

    The problem separates by rows of M, so all rows are updated together.
    With G = Γ1 Γ1ᵀ, minimizing over column j alone gives, exactly,
        M[:, j] = soft((Y Γ1ᵀ)[:, j] − Σ_{l≠j} M[:, l] G[l, j], λ/2) / G[j, j].
    Warm-started at the least-squares solution.
    """
    G = gamma1 @ gamma1.T
    YG = Y @ gamma1.T
    M = np.linalg.lstsq(gamma1.T, Y.T, rcond=None)[0].T
    for _ in range(_CD_MAX_SWEEPS):
        largest_step = 0.0
        for j in range(G.shape[0]):
            rho = YG[:, j] - M @ G[:, j] + M[:, j] * G[j, j]
            new = np.sign(rho) * np.maximum(np.abs(rho) - l1_lambda / 2.0, 0.0) / G[j, j]
            largest_step = max(largest_step, np.max(np.abs(new - M[:, j])))
            M[:, j] = new
        if largest_step <= _CD_TOL * max(np.max(np.abs(M)), 1e-300):
            return M
    raise RuntimeError(f"coordinate descent did not converge in {_CD_MAX_SWEEPS} sweeps")


def fit_edmd(gamma1, gamma2, state_dim, input_dim, dist_dim, l1_lambda, constant_index=None):
    """Solve the regularized least squares; return (A, B, D).

    Args:
        gamma1, gamma2: Γ1, Γ2 from build_snapshots, shape (N + m + d, K).
        state_dim: N, the lifted dimension.
        input_dim: m.
        dist_dim: d.
        l1_lambda: λ in Eq. (8), ≥ 0.
        constant_index: j of a constant feature ψ_j ≡ 1, whose row is then fixed
            at A[j] = e_jᵀ, B[j] = D[j] = 0 instead of fitted; None fits all rows.

    Returns:
        A (N, N), B (N, m), D (N, d).

    Raises:
        ValueError: If constant_index is out of range or ψ_j is not 1 in every
            snapshot of Γ1 and Γ2.
    """
    gamma1 = np.asarray(gamma1, dtype=float)
    gamma2 = np.asarray(gamma2, dtype=float)
    rows = state_dim + input_dim + dist_dim
    if gamma1.ndim != 2 or gamma1.shape[0] != rows or gamma2.shape != gamma1.shape:
        raise ValueError(f"gamma1 and gamma2 must both have shape ({rows}, K), "
                         f"got {gamma1.shape} and {gamma2.shape}")
    if not (np.all(np.isfinite(gamma1)) and np.all(np.isfinite(gamma2))):
        raise ValueError("gamma1 or gamma2 contains NaN or inf")
    if not (np.isfinite(l1_lambda) and l1_lambda >= 0):
        raise ValueError(f"l1_lambda must be non-negative and finite, got {l1_lambda}")

    fitted = np.arange(state_dim)
    if constant_index is not None:
        if not (isinstance(constant_index, (int, np.integer)) and 0 <= constant_index < state_dim):
            raise ValueError(f"constant_index must be in [0, {state_dim}), got {constant_index!r}")
        if not (np.all(gamma1[constant_index] == 1.0) and np.all(gamma2[constant_index] == 1.0)):
            raise ValueError(f"feature {constant_index} is not 1 in every snapshot; "
                             "it is not the constant monomial")
        fitted = np.delete(fitted, constant_index)

    Y = gamma2[fitted]
    if l1_lambda == 0:
        M_fitted = np.linalg.lstsq(gamma1.T, Y.T, rcond=None)[0].T
    else:
        M_fitted = _lasso_coordinate_descent(gamma1, Y, l1_lambda)
    M = np.zeros((state_dim, rows))
    M[fitted] = M_fitted
    if constant_index is not None:
        M[constant_index, constant_index] = 1.0
    return (M[:, :state_dim],
            M[:, state_dim:state_dim + input_dim],
            M[:, state_dim + input_dim:])
