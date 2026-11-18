"""Maximal admissible (positively invariant) set of a stable linear closed loop.

Terminal set for MPC (docs/koopman_mpc_study_guide.md §5.6; Gilbert & Tan,
IEEE TAC 1991):

    O∞ = { x : H A_clᵗ x ≤ h  for all t ≥ 0 }

is the set of states from which the terminal controller u = K x, with
A_cl = A + B K, keeps every constraint H x ≤ h satisfied forever (input
constraints enter as rows of H K). It is positively invariant and, for a Schur
A_cl and a bounded constraint set containing the origin in its interior, it is
a polytope found after finitely many steps.

Algorithm: start from {H x ≤ h}; at step t, add the rows H A_clᵗ; stop when
no new row can be violated anywhere in the current set (one LP per row).
"""
import numpy as np
from scipy.optimize import linprog

_DEFAULT_MAX_STEPS = 200
_REDUNDANCY_TOL = 1e-9   # a row is redundant if its LP maximum exceeds h by at most this


def maximal_admissible_set(A_cl, H, h, max_steps: int = _DEFAULT_MAX_STEPS):
    """Return (H_f, h_f) with O∞ = {x : H_f x ≤ h_f}.

    Args:
        A_cl: (n, n) closed-loop matrix; must be Schur (spectral radius < 1).
        H: (r, n) constraint rows, e.g. [E; −E; K; −K] for state and input bounds.
        h: (r,) right-hand side; must be > 0 so the origin is strictly inside.
        max_steps: Upper bound on the number of propagation steps t.

    Returns:
        H_f (r (t*+1), n), h_f (r (t*+1),), where t* is the step at which the
        set stopped growing.

    Raises:
        ValueError: On non-Schur A_cl, non-positive h or bad shapes.
        RuntimeError: If the set has not converged after max_steps.
    """
    A_cl = np.asarray(A_cl, dtype=float)
    H = np.asarray(H, dtype=float)
    h = np.asarray(h, dtype=float)
    n = A_cl.shape[0]
    if A_cl.shape != (n, n) or H.ndim != 2 or H.shape[1] != n or h.shape != (H.shape[0],):
        raise ValueError(f"shapes must be A_cl (n, n), H (r, n), h (r,); got "
                         f"{A_cl.shape}, {H.shape}, {h.shape}")
    if not (np.all(np.isfinite(A_cl)) and np.all(np.isfinite(H)) and np.all(np.isfinite(h))):
        raise ValueError("A_cl, H and h must be finite")
    if np.max(np.abs(np.linalg.eigvals(A_cl))) >= 1.0:
        raise ValueError("A_cl must be Schur stable")
    if np.any(h <= 0):
        raise ValueError("h must be strictly positive (origin inside the constraint set)")

    H_f, h_f = H.copy(), h.copy()
    A_power = np.eye(n)
    free = [(None, None)] * n
    for _ in range(max_steps):
        A_power = A_cl @ A_power
        new_rows = H @ A_power
        converged = True
        for row, bound in zip(new_rows, h):
            result = linprog(-row, A_ub=H_f, b_ub=h_f, bounds=free, method="highs")
            # Unbounded (status 3) or a maximum above the bound: the row still cuts the set.
            if result.status != 0 or -result.fun > bound + _REDUNDANCY_TOL:
                converged = False
                break
        if converged:
            return H_f, h_f
        H_f = np.vstack([H_f, new_rows])
        h_f = np.concatenate([h_f, h])
    raise RuntimeError(f"maximal admissible set did not converge in {max_steps} steps")
