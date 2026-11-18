"""Dense (condensed) QP formulation of linear MPC, following Korda & Mezić [25].

K-MPC, Eqs. (15)-(16) (derivation in docs/mpc_guide.md §2.1-2.2):

    min  Σ_{k=1}^{np-1} ‖C z_k − r_k‖²_Q + ‖C z_np − r_np‖²_P + Σ_{k=0}^{np-1} ν_kᵀ R ν_k
    s.t. z_{k+1} = A z_k + B ν_k + D v̄_w,   z_0 given
         u_lb ≤ ν_k ≤ u_ub,                  k = 0..np-1
         h_lb ≤ E z_k ≤ h_ub,                k = 1..np

The z_0 cost term is constant and dropped; z_0 is measured, so it is not
constrained either. Eliminating the states with Z = [z_1; ...; z_np] and
V = [ν_0; ...; ν_{np-1}],

    Z = Φ z_0 + Γ V + Λ v̄_w,

turns the problem into  min ½ Vᵀ H V + fᵀ V  s.t. linear inequalities, where
H is fixed, f is affine in (z_0, v̄_w, r) and the constraint bounds are affine
in (z_0, v̄_w). The paper uses P = Q and r = 0; E selects the constrained
combinations of the lifted state.

The output reference r_k is the costed output the MPC drives towards. It is
needed whenever zero physical error is not C z = 0, e.g. for a standardized
state z = (e − μ)/σ, where e = 0 is C z = −μ/σ, and it carries a time-varying
preview (e.g. deck motion) without changing H.

Units and frames are those of the model (here: standardized lifted error
state, physical perturbation input and wind); this module is agnostic of them.
"""
from dataclasses import dataclass
import time

import daqp
import numpy as np

_SYMMETRY_TOL = 1e-9


def _as_finite_matrix(name: str, value, shape: tuple[int, int]) -> np.ndarray:
    """Return `value` as a float array of `shape`, or raise ValueError."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def _check_symmetric(name: str, M: np.ndarray) -> None:
    if not np.allclose(M, M.T, atol=_SYMMETRY_TOL, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")


def _check_psd(name: str, M: np.ndarray) -> None:
    _check_symmetric(name, M)
    min_eig = np.linalg.eigvalsh(M).min()
    if min_eig < -_SYMMETRY_TOL * max(1.0, np.abs(M).max()):
        raise ValueError(f"{name} must be positive semidefinite, min eigenvalue {min_eig:.3e}")


@dataclass(frozen=True)
class CondensedMPC:
    """Horizon-np prediction and cost matrices of the condensed QP.

    Shapes, with N lifted states, m inputs, d disturbances, n_E constrained
    outputs, n costed outputs: Phi (np N, N), Gamma (np N, np m),
    Lambda (np N, d), H (np m, np m), F_z (np m, N), F_w (np m, d),
    F_r (np m, np n), G (np n_E, np m).
    """

    Phi: np.ndarray
    Gamma: np.ndarray
    Lambda: np.ndarray
    H: np.ndarray
    F_z: np.ndarray  # f = F_z z_0 + F_w v̄_w + F_r r̄,  r̄ = [r_1; ...; r_np]
    F_w: np.ndarray
    F_r: np.ndarray
    G: np.ndarray    # Ē Γ, state-constraint matrix on V
    E_bar: np.ndarray
    horizon: int

    @property
    def num_vars(self) -> int:
        return self.H.shape[0]

    def predict(self, z0, V, v_w) -> np.ndarray:
        """States z_1..z_np, shape (np, N), from z_0 (N,), V (np m,), v̄_w (d,)."""
        Z = self.Phi @ z0 + self.Gamma @ V + self.Lambda @ v_w
        return Z.reshape(self.horizon, -1)

    def linear_term(self, z0, v_w, r=None) -> np.ndarray:
        """f in ½ Vᵀ H V + fᵀ V for initial lifted state z_0 and mean wind v̄_w.

        r: output reference, (n,) for a constant target or (np, n) for
        r_1..r_np; default 0 (the paper's cost).
        """
        f = self.F_z @ z0 + self.F_w @ v_w
        if r is None:
            return f
        n = self.F_r.shape[1] // self.horizon
        r = np.broadcast_to(np.asarray(r, dtype=float), (self.horizon, n))
        if not np.all(np.isfinite(r)):
            raise ValueError("r contains NaN or inf")
        return f + self.F_r @ r.ravel()

    def state_bounds(self, z0, v_w, h_lb, h_ub) -> tuple[np.ndarray, np.ndarray]:
        """Bounds (g_lb, g_ub) on G V equivalent to h_lb ≤ E z_k ≤ h_ub, k = 1..np.

        h_lb, h_ub: (n_E,) for constant bounds, or (np, n_E) for per-step
        bounds such as the tightened ones of K-SMPC, Eq. (19a). ±inf allowed.
        """
        n_e = self.E_bar.shape[0] // self.horizon
        offset = self.E_bar @ (self.Phi @ z0 + self.Lambda @ v_w)
        bounds = []
        for name, h in (("h_lb", h_lb), ("h_ub", h_ub)):
            h = np.broadcast_to(np.asarray(h, dtype=float), (self.horizon, n_e))
            if np.any(np.isnan(h)):
                raise ValueError(f"{name} contains NaN")
            bounds.append(h.ravel() - offset)
        return bounds[0], bounds[1]


def _validate_cost(C, Q, R, P, E, N: int, m: int):
    """Shape- and definiteness-checked (C, Q, R, P, E); P defaults to Q, E to C."""
    C = np.asarray(C, dtype=float)
    n = C.shape[0] if C.ndim == 2 else -1
    C = _as_finite_matrix("C", C, (n, N))
    Q = _as_finite_matrix("Q", Q, (n, n))
    R = _as_finite_matrix("R", R, (m, m))
    P = Q if P is None else _as_finite_matrix("P", P, (n, n))
    E = C if E is None else np.asarray(E, dtype=float)
    E = _as_finite_matrix("E", E, (E.shape[0] if E.ndim == 2 else -1, N))
    _check_psd("Q", Q)
    _check_psd("P", P)
    _check_symmetric("R", R)
    try:
        np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        raise ValueError("R must be positive definite") from None
    return C, Q, R, P, E


def _assemble(Phi, Gamma, Lambda, C, Q, R, P, E, horizon: int) -> CondensedMPC:
    """Cost and constraint matrices of the condensed QP from the prediction Z = Φz₀ + ΓV + Λv."""
    n = C.shape[0]
    C_bar = np.kron(np.eye(horizon), C)
    Q_bar = np.kron(np.eye(horizon), Q)
    Q_bar[-n:, -n:] = P
    W = C_bar.T @ Q_bar @ C_bar
    GtW = Gamma.T @ W
    H = 2.0 * (GtW @ Gamma + np.kron(np.eye(horizon), R))
    H = 0.5 * (H + H.T)  # remove round-off asymmetry
    E_bar = np.kron(np.eye(horizon), E)
    return CondensedMPC(
        Phi=Phi, Gamma=Gamma, Lambda=Lambda, H=H,
        F_z=2.0 * GtW @ Phi, F_w=2.0 * GtW @ Lambda,
        F_r=-2.0 * Gamma.T @ C_bar.T @ Q_bar,  # from −2 r̄ᵀ Q̄ C̄ Γ V
        G=E_bar @ Gamma, E_bar=E_bar, horizon=horizon,
    )


def condense(A, B, D, C, Q, R, horizon, P=None, E=None) -> CondensedMPC:
    """Build the condensed QP of K-MPC, Eqs. (15)-(16), see module docstring.

    Args:
        A, B, D: Koopman model, ψ_{k+1} = A ψ_k + B u_k + D v_k, Eq. (5).
        C: (n, N), maps the lifted state to the costed output.
        Q: (n, n) symmetric PSD stage weight on C z_k − r_k.
        R: (m, m) symmetric PD input weight.
        horizon: n_p ≥ 1.
        P: (n, n) symmetric PSD terminal weight on C z_np; default Q (paper).
        E: (n_E, N) state-constraint selector; default C.
    """
    if not (isinstance(horizon, (int, np.integer)) and horizon >= 1):
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    horizon = int(horizon)
    A = np.asarray(A, dtype=float)
    N = A.shape[0] if A.ndim == 2 else -1
    A = _as_finite_matrix("A", A, (N, N))
    B = np.asarray(B, dtype=float)
    m = B.shape[1] if B.ndim == 2 else -1
    B = _as_finite_matrix("B", B, (N, m))
    D = np.asarray(D, dtype=float)
    d = D.shape[1] if D.ndim == 2 else -1
    D = _as_finite_matrix("D", D, (N, d))
    C, Q, R, P, E = _validate_cost(C, Q, R, P, E, N, m)

    # A^k B for k = 0..np-1, and the rows of Φ, Λ for z_1..z_np.
    AkB = [B]
    for _ in range(horizon - 1):
        AkB.append(A @ AkB[-1])
    Phi_rows, Lambda_rows = [A], [D]
    for _ in range(horizon - 1):
        Phi_rows.append(A @ Phi_rows[-1])
        Lambda_rows.append(A @ Lambda_rows[-1] + D)  # Σ_{j<k} A^j D
    Gamma = np.zeros((horizon * N, horizon * m))
    for k in range(horizon):        # row block: z_{k+1}
        for j in range(k + 1):      # column block: ν_j
            Gamma[k * N:(k + 1) * N, j * m:(j + 1) * m] = AkB[k - j]
    return _assemble(np.vstack(Phi_rows), Gamma, np.vstack(Lambda_rows), C, Q, R, P, E, horizon)


def condense_ltv(As, Bs, Ds, C, Q, R, P=None, E=None) -> CondensedMPC:
    """Condensed QP for a time-varying model over one horizon, n_p = len(As).

        z_{i+1} = A_i z_i + B_i ν_i + D_i v,   i = 0..n_p−1,  v constant over the horizon

    Same cost, constraints and returned CondensedMPC as `condense`; only the
    prediction matrices change. With Φ(i, j) = A_{i−1} ⋯ A_j (Φ(j, j) = I):
        Φ rows  Φ(i, 0),   Γ block (i, j) = Φ(i, j+1) B_j for j < i,
        Λ rows  Σ_{j<i} Φ(i, j+1) D_j.
    With every A_i = A, B_i = B, D_i = D this is `condense`.

    An affine term c_i (e.g. a reference residual) is the case D_i = c_i[:, None]
    and v = [1].

    Args:
        As (n_p, N, N), Bs (n_p, N, m), Ds (n_p, N, d): model at steps 0..n_p−1.
        C, Q, R, P, E: as in `condense`.
    """
    As = np.asarray(As, dtype=float)
    if As.ndim != 3 or As.shape[0] < 1 or As.shape[1] != As.shape[2]:
        raise ValueError(f"As must have shape (n_p, N, N) with n_p ≥ 1, got {As.shape}")
    horizon, N = As.shape[0], As.shape[1]
    Bs, Ds = np.asarray(Bs, dtype=float), np.asarray(Ds, dtype=float)
    m = Bs.shape[2] if Bs.ndim == 3 else -1
    d = Ds.shape[2] if Ds.ndim == 3 else -1
    for name, arr, shape in (("As", As, (horizon, N, N)), ("Bs", Bs, (horizon, N, m)),
                             ("Ds", Ds, (horizon, N, d))):
        if arr.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or inf")
    C, Q, R, P, E = _validate_cost(C, Q, R, P, E, N, m)

    # Row block i holds z_{i+1}: propagate each earlier column block by A_i.
    Phi_rows, Lambda_rows = [], []
    Gamma = np.zeros((horizon * N, horizon * m))
    Phi_i, Lambda_i = np.eye(N), np.zeros((N, d))
    for i in range(horizon):
        Phi_i = As[i] @ Phi_i
        Lambda_i = As[i] @ Lambda_i + Ds[i]
        Phi_rows.append(Phi_i)
        Lambda_rows.append(Lambda_i)
        rows = slice(i * N, (i + 1) * N)
        if i > 0:
            Gamma[rows, :i * m] = As[i] @ Gamma[(i - 1) * N:i * N, :i * m]
        Gamma[rows, i * m:(i + 1) * m] = Bs[i]
    return _assemble(np.vstack(Phi_rows), Gamma, np.vstack(Lambda_rows), C, Q, R, P, E, horizon)


@dataclass(frozen=True)
class QPResult:
    """Solver output. `x` is meaningful only when `solved` is True.

    exitflag is DAQP's status code (1 optimal, -1 infeasible; full list at
    https://darnstrom.github.io/daqp/parameters). solve_time is the wall-clock
    time of the DAQP call, s, setup (factorizing H) included; DAQP's own
    info["solve_time"] counts only the active-set iterations.
    """

    x: np.ndarray
    solved: bool
    exitflag: int
    iterations: int
    solve_time: float


def solve_qp(H, f, lb=None, ub=None, G=None, g_lb=None, g_ub=None, x0=None) -> QPResult:
    """Solve  min ½ xᵀ H x + fᵀ x  s.t.  lb ≤ x ≤ ub,  g_lb ≤ G x ≤ g_ub  with DAQP.

    Any bound may be None (unbounded) or contain ±inf. x0 warm-starts the
    active set. Infeasibility is reported in the result, never raised, so the
    caller can fall back and log it.
    """
    f = np.asarray(f, dtype=float)
    n = f.shape[0] if f.ndim == 1 else -1
    H = _as_finite_matrix("H", H, (n, n))
    f = np.ascontiguousarray(f)
    if not np.all(np.isfinite(f)):
        raise ValueError("f contains NaN or inf")
    _check_symmetric("H", H)
    if G is None:
        G = np.zeros((0, n))
    G = np.asarray(G, dtype=float)
    G = _as_finite_matrix("G", G, (G.shape[0] if G.ndim == 2 else -1, n))
    n_g = G.shape[0]

    def bound(name, value, size, fill):
        arr = np.full(size, fill) if value is None else np.asarray(value, dtype=float)
        if arr.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},), got {arr.shape}")
        if np.any(np.isnan(arr)):
            raise ValueError(f"{name} contains NaN")
        return arr

    lower = np.concatenate([bound("lb", lb, n, -np.inf), bound("g_lb", g_lb, n_g, -np.inf)])
    upper = np.concatenate([bound("ub", ub, n, np.inf), bound("g_ub", g_ub, n_g, np.inf)])
    primal_start = None if x0 is None else np.ascontiguousarray(bound("x0", x0, n, 0.0))
    start = time.perf_counter()
    x, _, exitflag, info = daqp.solve(
        np.ascontiguousarray(H), f, np.ascontiguousarray(G), upper, lower,
        primal_start=primal_start,
    )
    elapsed = time.perf_counter() - start
    return QPResult(x=np.asarray(x), solved=exitflag == 1, exitflag=int(exitflag),
                    iterations=int(info["iterations"]), solve_time=elapsed)
