"""Glue between K-MPC and the closed-loop simulator.

KoopmanMPC works on the tracking error e_x and knows nothing about the 13-state
or the simulator; simulate() takes any Augmentation u_d(t, x, x_ref) and knows
nothing about MPC. KMPCAugmentation connects the two and records every solve,
so the run record can show what the MPC did, including fallbacks.
"""
from dataclasses import dataclass, fields

import numpy as np

from controllers.kmpc import KoopmanMPC, MPCStep
from datagen.collect import tracking_error


@dataclass(frozen=True)
class MPCLog:
    """One row per K-MPC solve (every dt_augment).

    Attributes:
        t: Solve time stamps, s, shape (K,).
        e_x: Measured tracking error, shape (K, n), units in controllers.kmpc.
        u_d: Applied perturbation, shape (K, m); zeros where solved is False.
        solved: QP solved, shape (K,); False means fallback to the LQR alone.
        exitflag: DAQP status, shape (K,).
        iterations: Active-set iterations, shape (K,).
        solve_time: Wall-clock time of each K-MPC step, s, shape (K,).
        planned_inputs: ν_0..ν_{np-1}, shape (K, np, m).
        predicted_error: e_1..e_np for the plan, shape (K, np, n).
    """

    t: np.ndarray
    e_x: np.ndarray
    u_d: np.ndarray
    solved: np.ndarray
    exitflag: np.ndarray
    iterations: np.ndarray
    solve_time: np.ndarray
    planned_inputs: np.ndarray
    predicted_error: np.ndarray

    def save(self, path) -> None:
        """Write every field to path (.npz)."""
        np.savez(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path) -> "MPCLog":
        """Inverse of save."""
        with np.load(path, allow_pickle=False) as data:
            return cls(**{f.name: data[f.name] for f in fields(cls)})


class KMPCAugmentation:
    """Augmentation for simulate(): u_d = K-MPC(e_x(x, x_ref)), every solve recorded."""

    def __init__(self, mpc: KoopmanMPC):
        self.mpc = mpc
        self._times: list[float] = []
        self._errors: list[np.ndarray] = []
        self._steps: list[MPCStep] = []

    def __call__(self, t: float, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """u_d, shape (4,), for the 13-state x and 10-state x_ref at time t (s)."""
        e_x = tracking_error(x[None, :], x_ref[None, :])[0]
        step = self.mpc.solve(e_x)
        self._times.append(float(t))
        self._errors.append(e_x)
        self._steps.append(step)
        return step.u_d

    @property
    def fallback_count(self) -> int:
        return sum(not step.solved for step in self._steps)

    def log(self) -> MPCLog:
        """All solves so far; raises if there were none."""
        if not self._steps:
            raise ValueError("no K-MPC solves recorded")
        steps = self._steps
        return MPCLog(
            t=np.array(self._times),
            e_x=np.array(self._errors),
            u_d=np.array([s.u_d for s in steps]),
            solved=np.array([s.solved for s in steps]),
            exitflag=np.array([s.exitflag for s in steps]),
            iterations=np.array([s.iterations for s in steps]),
            solve_time=np.array([s.solve_time for s in steps]),
            planned_inputs=np.array([s.planned_inputs for s in steps]),
            predicted_error=np.array([s.predicted_error for s in steps]),
        )
