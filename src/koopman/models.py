"""Learned Koopman error models.

Model 1, Eq. (5):  ψ_{k+1} = A ψ_k + B u_{d,k} + D v_{w,k}
Model 2, Eq. (6):  ψ_{k+1} = A ψ_k + B (u_{d,k} + u_{res,k}) + D v_{w,k},
                   u_res = [0, 0, 0, f_res], f_res = f_c − f_r
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class KoopmanModel:
    A: np.ndarray
    B: np.ndarray
    D: np.ndarray
    C: np.ndarray  # projects lifted state back to the error state
    use_residual: bool = False  # False -> Model 1, True -> Model 2

    def __post_init__(self) -> None:
        N = self.A.shape[0]
        if self.A.shape != (N, N):
            raise ValueError(f"A must be square, got {self.A.shape}")
        for name, M in (("B", self.B), ("D", self.D)):
            if M.ndim != 2 or M.shape[0] != N:
                raise ValueError(f"{name} must have {N} rows, got {M.shape}")
        if self.C.ndim != 2 or self.C.shape[1] != N:
            raise ValueError(f"C must have {N} columns, got {self.C.shape}")
        for name in ("A", "B", "D", "C"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name} contains NaN or inf")

    def predict(self, psi, u_d, v_w, u_res=None):
        """One step of Eq. (5) (or (6) with u_res): ψ (N,), u_d (m,), v_w (d,) -> ψ_k+1 (N,)."""
        if self.use_residual and u_res is None:
            raise ValueError("Model 2 needs u_res")
        if not self.use_residual and u_res is not None:
            raise ValueError("Model 1 takes no u_res")
        u = np.asarray(u_d, dtype=float)
        if u_res is not None:
            u = u + np.asarray(u_res, dtype=float)
        return self.A @ np.asarray(psi, dtype=float) + self.B @ u + self.D @ np.asarray(v_w, dtype=float)

    def save(self, path):
        """Write A, B, D, C and use_residual to path (.npz)."""
        np.savez(path, A=self.A, B=self.B, D=self.D, C=self.C,
                 use_residual=np.array(self.use_residual))

    @classmethod
    def load(cls, path):
        """Inverse of save."""
        with np.load(path, allow_pickle=False) as data:
            return cls(A=data["A"], B=data["B"], D=data["D"], C=data["C"],
                       use_residual=bool(data["use_residual"]))
