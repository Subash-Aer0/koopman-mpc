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

    def predict(self, psi, u_d, v_w, u_res=None):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError
