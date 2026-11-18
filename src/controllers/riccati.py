"""Time-varying LQR: the backward Riccati recursion, for LTV MPC terminal ingredients.

For x_{k+1} = A_k x_k + B_k u_k and the cost Σ x_kᵀ Q x_k + u_kᵀ R u_k + x_Kᵀ P_end x_K,
the cost-to-go is x_kᵀ P_k x_k with (finite-horizon LQ problem by dynamic programming,
e.g. Bertsekas, Dynamic Programming and Optimal Control, Vol. I)

    K_k = −(R + B_kᵀ P_{k+1} B_k)⁻¹ B_kᵀ P_{k+1} A_k
    P_k = Q + A_kᵀ P_{k+1} (A_k + B_k K_k),           P_K = P_end

and u_k = K_k x_k is optimal. With A_k = A, B_k = B and K → ∞, P_0 tends to the
DARE solution. For a periodic system, P_k at samples far before K tends to the
periodic solution, independent of P_end: the LTV counterpart of the DARE, and
why the MPC can extend its grid past the run and recurse back over it.

Units are those of the model; this module is agnostic of them.
"""
import numpy as np


def time_varying_riccati(As, Bs, Q, R, P_end) -> tuple[np.ndarray, np.ndarray]:
    """Backward Riccati recursion over K steps.

    Args:
        As (K, n, n), Bs (K, n, m): model at steps 0..K−1.
        Q (n, n): symmetric PSD stage weight.
        R (m, m): symmetric PD input weight.
        P_end (n, n): symmetric PSD weight on x_K.

    Returns:
        P (K+1, n, n), symmetric PSD cost-to-go matrices P_0..P_K (P_K = P_end),
        and K (K, m, n), the optimal gains K_0..K_{K−1}.

    Raises:
        ValueError: On wrong shapes, non-finite values, or weights that are not
            symmetric (semi)definite.
    """
    As, Bs = np.asarray(As, dtype=float), np.asarray(Bs, dtype=float)
    if As.ndim != 3 or As.shape[0] < 1 or As.shape[1] != As.shape[2]:
        raise ValueError(f"As must have shape (K, n, n) with K ≥ 1, got {As.shape}")
    steps, n = As.shape[0], As.shape[1]
    if Bs.ndim != 3 or Bs.shape[:2] != (steps, n):
        raise ValueError(f"Bs must have shape ({steps}, {n}, m), got {Bs.shape}")
    m = Bs.shape[2]
    Q, R, P_end = (np.asarray(M, dtype=float) for M in (Q, R, P_end))
    for name, M, size, strict in (("Q", Q, n, False), ("R", R, m, True), ("P_end", P_end, n, False)):
        if M.shape != (size, size) or not np.all(np.isfinite(M)):
            raise ValueError(f"{name} must be a finite {size}x{size} matrix, got shape {M.shape}")
        if not np.allclose(M, M.T):
            raise ValueError(f"{name} must be symmetric")
        min_eig = np.linalg.eigvalsh(M).min()
        if (strict and min_eig <= 0) or (not strict and min_eig < -1e-12):
            raise ValueError(f"{name} must be positive {'definite' if strict else 'semidefinite'}")
    if not (np.all(np.isfinite(As)) and np.all(np.isfinite(Bs))):
        raise ValueError("As and Bs must be finite")

    P = np.empty((steps + 1, n, n))
    K = np.empty((steps, m, n))
    P[steps] = P_end
    for k in range(steps - 1, -1, -1):
        A, B, P_next = As[k], Bs[k], P[k + 1]
        K[k] = -np.linalg.solve(R + B.T @ P_next @ B, B.T @ P_next @ A)
        P_k = Q + A.T @ P_next @ (A + B @ K[k])
        P[k] = 0.5 * (P_k + P_k.T)   # remove round-off asymmetry
    return P, K
