"""Persistency of excitation of the input, via the rank of block-Hankel matrices.

For an input sequence u_1..u_T (l channels), the depth-L block-Hankel matrix

    H_L(u) = [[u_1, u_2, ..., u_{T−L+1}],
              [u_2, u_3, ..., u_{T−L+2}],
              ...
              [u_L, ..., u_T]]                 ∈ R^{lL × (T−L+1)}

has full row rank lL exactly when u is persistently exciting (PE) of order L
[LN-DDC §2.2.3, Eq. (5)]. Willems' lemma: for a controllable linear system
with n states, PE of order n + 1 guarantees rank [X; U] = n + l, i.e. the
regression for [A B] has a unique solution.

Several short experiments are combined in a mosaic Hankel matrix
[H_L(u⁽¹⁾) ... H_L(u⁽ʳ⁾)] (collective PE, [LN-DDC §4.5]); windows never
cross an episode boundary. For a nonlinear (lifted) model the lemma gives no
guarantee, so the rank of the lifted regressor itself must also be checked
(koopman.preprocessing.regressor_diagnostics).

Ranks are judged on singular values, not the integer rank alone: the smallest
singular value relative to the largest says how close the data are to losing
excitation.
"""
import numpy as np

_RANK_RTOL = 1e-9   # singular values below this fraction of the largest count as zero


def hankel_matrix(u, depth: int) -> np.ndarray:
    """Depth-L block-Hankel matrix of u (T, l): shape (l L, T − L + 1)."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 2:
        raise ValueError(f"u must have shape (T, l), got {u.shape}")
    if not np.all(np.isfinite(u)):
        raise ValueError("u contains NaN or inf")
    T = u.shape[0]
    if not (isinstance(depth, (int, np.integer)) and 1 <= depth <= T):
        raise ValueError(f"depth must be an integer in [1, {T}], got {depth!r}")
    return np.vstack([u[i:T - depth + 1 + i].T for i in range(depth)])


def persistency_of_excitation(episodes, depth: int) -> dict:
    """Collective PE test of order `depth` over a list of input episodes.

    Args:
        episodes: List of arrays (T_i, l), one per experiment; episodes shorter
            than `depth` contribute no columns.
        depth: L, the PE order tested.

    Returns:
        {"order", "rows" (= l L), "columns", "rank", "persistently_exciting",
         "sigma_min_rel" (σ_min / σ_max), "min_samples_single_experiment"
         (= (l + 1) L − 1, the length one experiment would need at least)}.
    """
    blocks = [hankel_matrix(u, depth) for u in episodes if len(u) >= depth]
    if not blocks:
        raise ValueError(f"no episode is at least {depth} samples long")
    H = np.hstack(blocks)
    rows, columns = H.shape
    l = rows // depth
    s = np.linalg.svd(H, compute_uv=False)
    rank = int(np.sum(s > _RANK_RTOL * s[0]))
    return {
        "order": int(depth),
        "rows": int(rows),
        "columns": int(columns),
        "rank": rank,
        "persistently_exciting": rank == rows,
        "sigma_min_rel": float(s[min(rows, columns) - 1] / s[0]) if columns >= rows else 0.0,
        "min_samples_single_experiment": int((l + 1) * depth - 1),
    }
