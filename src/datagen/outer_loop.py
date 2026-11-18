"""Identification data for the outer-loop Koopman model: dithered flights, total commands.

The flights are flown by the baseline MPC (it holds commands at 10 Hz, the
model's sample time) with a uniform random dither added to its command. The
dither is what makes the command not a function of the state: without it the
regression could only recover the closed loop ([LN-DDC §2.2.2]).

Each episode stores, at the 10 Hz command instants:
    x9      [p, v, η] world NED, ZYX Euler (m, m/s, rad)      (T, 9)
    omega   measured body rates, body FRD, rad/s               (T, 3)
    u       applied total command [ω (rad/s), c (m/s²)]        (T, 4)
    v_w     wind, world NED, m/s, held over the sample          (T, 3)
Sample k's command and wind act over [t_k, t_k+1), so (x_k, u_k, v_w,k) -> x_k+1.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from controllers.linear_mpc import outer_state
from dynamics import dynamics as rigid_body
from simulation.closed_loop import SimulationLog

DATASET_FILE = "outer_loop_dataset.npz"
FIELDS = ("x9", "omega", "u", "v_w")


class DitheredController:
    """controller(t, x) + d_k, d_k ~ U(−a, a) per channel, redrawn every period, then clipped."""

    def __init__(self, controller, bounds, u_lb, u_ub, rng: np.random.Generator):
        self.controller = controller
        self.period = controller.period
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.shape != (4,) or np.any(self.bounds < 0):
            raise ValueError(f"dither bounds must be 4 non-negative numbers, got {bounds}")
        self.u_lb = np.asarray(u_lb, dtype=float)
        self.u_ub = np.asarray(u_ub, dtype=float)
        self.rng = rng

    def __call__(self, t: float, x) -> np.ndarray:
        u = self.controller(t, x) + self.rng.uniform(-self.bounds, self.bounds)
        return np.clip(u, self.u_lb, self.u_ub)


@dataclass
class Episode:
    name: str
    x9: np.ndarray
    omega: np.ndarray
    u: np.ndarray
    v_w: np.ndarray


def episode_from_log(name: str, log: SimulationLog) -> Episode:
    """Episode at the command instants; requires dt_log == command period."""
    if not np.isclose(log.timing.dt_log, log.timing.dt_control):
        raise ValueError("log the episode at the command period so u_k is the held command")
    x9 = np.array([outer_state(x) for x in log.x])
    return Episode(name=name, x9=x9, omega=log.x[:, rigid_body.RATE].copy(),
                   u=log.u_cmd.copy(), v_w=log.v_w.copy())


def save_episodes(directory, splits: dict[str, list[Episode]]) -> Path:
    """Write every split as concatenated arrays plus episode boundaries (no pickle)."""
    arrays = {}
    for split, episodes in splits.items():
        lengths = np.array([len(ep.u) for ep in episodes])
        arrays[f"{split}.lengths"] = lengths
        arrays[f"{split}.names"] = np.array([ep.name for ep in episodes])
        for field in FIELDS:
            arrays[f"{split}.{field}"] = np.concatenate([getattr(ep, field) for ep in episodes])
    path = Path(directory) / DATASET_FILE
    np.savez(path, **arrays)
    return path


def load_episodes(directory) -> dict[str, list[Episode]]:
    path = Path(directory)
    if path.is_dir():
        path = path / DATASET_FILE
    splits = {}
    with np.load(path, allow_pickle=False) as data:
        for split in sorted({key.split(".")[0] for key in data.files}):
            bounds = np.concatenate([[0], np.cumsum(data[f"{split}.lengths"])])
            names = data[f"{split}.names"]
            splits[split] = [
                Episode(name=str(names[i]),
                        **{f: data[f"{split}.{f}"][bounds[i]:bounds[i + 1]] for f in FIELDS})
                for i in range(len(names))
            ]
    return splits
