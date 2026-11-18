"""EDMD least squares with translation invariance imposed on the position states.

The quadrotor's dynamics do not depend on where it is, so in the lifted model
    z⁺ = A z + B u + D w
position may only feed itself: p_i⁺ = p_i + (terms in other features), and no
other feature may depend on p. Columns of A for the position features are
therefore fixed to the unit vectors e_i and only the other columns are fitted.

Why impose it rather than learn it: if the training flights barely vary a
position channel (e.g. every flight at the same altitude), that channel is
nearly collinear with the constant feature, the unconstrained fit splits a bias
between them arbitrarily, and the model then predicts a phantom force at any
other position — an identifiability failure ([LN-DDC] Question 1) that
validation at the training position does not reveal.

Requires the lifting order of SelectiveMonomialLifting: the state first, so
feature i is state i for i < n.
"""
import numpy as np


def fit_translation_invariant(gamma1, gamma2, lifted_dim: int, input_dim: int,
                              dist_dim: int, position_indices, constant_index: int):
    """Least squares for [A B D] with A[:, p] = e_p and the constant row fixed.

    Args:
        gamma1, gamma2: Snapshot matrices from build_snapshots, (N + m + d, K).
        lifted_dim: N.
        input_dim, dist_dim: m, d.
        position_indices: Feature indices of the position states (< N).
        constant_index: Index of the constant feature ψ_c ≡ 1 (row fixed to e_cᵀ).

    Returns:
        A (N, N), B (N, m), D (N, d).
    """
    gamma1 = np.asarray(gamma1, dtype=float)
    gamma2 = np.asarray(gamma2, dtype=float)
    rows = lifted_dim + input_dim + dist_dim
    if gamma1.shape[0] != rows or gamma2.shape != gamma1.shape:
        raise ValueError(f"gamma1 and gamma2 must both have {rows} rows and equal shapes")
    if not (np.all(np.isfinite(gamma1)) and np.all(np.isfinite(gamma2))):
        raise ValueError("gamma1 or gamma2 contains NaN or inf")
    positions = sorted(int(i) for i in position_indices)
    if not positions or any(not 0 <= i < lifted_dim for i in positions) \
            or constant_index in positions:
        raise ValueError(f"position_indices must be lifted features other than the constant, "
                         f"got {positions}")
    if not (np.all(gamma1[constant_index] == 1.0) and np.all(gamma2[constant_index] == 1.0)):
        raise ValueError(f"feature {constant_index} is not the constant 1 in every snapshot")

    free_cols = [j for j in range(rows) if j not in positions]
    fitted_rows = [i for i in range(lifted_dim) if i != constant_index]
    # Remove the known position contribution, then regress on the remaining columns.
    target = gamma2[fitted_rows].copy()
    for i in positions:
        target[fitted_rows.index(i)] -= gamma1[i]
    M_free = np.linalg.lstsq(gamma1[free_cols].T, target.T, rcond=None)[0].T

    M = np.zeros((lifted_dim, rows))
    M[np.ix_(fitted_rows, free_cols)] = M_free
    for i in positions:
        M[i, i] = 1.0
    M[constant_index] = 0.0
    M[constant_index, constant_index] = 1.0
    return M[:, :lifted_dim], M[:, lifted_dim:lifted_dim + input_dim], M[:, lifted_dim + input_dim:]
