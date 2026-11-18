"""Dense (condensed) QP formulation of linear MPC, following Korda & Mezić [25]."""


def condense(A, B, D, C, Q, R, horizon):
    """Build prediction matrices so the MPC becomes a QP in ν_0..ν_{np-1}."""
    raise NotImplementedError


def solve_qp(H, f, A_ineq, b_ineq):
    raise NotImplementedError
