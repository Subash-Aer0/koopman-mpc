"""Load the hover linear-MPC config (configs/controllers/hover_mpc.toml) and build the MPC.

Kept apart from builders.py so the existing LQR / K-MPC loaders are untouched.
"""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from controllers.hover_model import HoverModel
from controllers.linear_mpc import (
    LinearizedReference, LinearMPC, LinearMPCSettings, PositionDerivatives,
)
from datagen.trajectories import SetpointReference
from dynamics.params import QuadParams

from .builders import load_toml


def _vector(section: dict, key: str, size: int) -> np.ndarray:
    if key not in section:
        raise ValueError(f"[hover_mpc] missing '{key}'")
    value = np.asarray(section[key], dtype=float)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f"[hover_mpc] {key} must be {size} finite numbers, got {section[key]}")
    return value


@dataclass(frozen=True)
class HoverMPCConfig:
    """Everything needed to build a LinearMPC, plus its config path for the run record."""

    sample_time: float
    yaw: float
    wind_mean: np.ndarray
    settings: LinearMPCSettings
    config_path: Path

    def with_terminal(self, terminal: str) -> "HoverMPCConfig":
        """The same config with another terminal mode (for ablations)."""
        return replace(self, settings=replace(self.settings, terminal=terminal))

    def with_horizon(self, horizon: int) -> "HoverMPCConfig":
        """The same config with another horizon N (for ablations)."""
        return replace(self, settings=replace(self.settings, horizon=horizon))

    def make_controller(self, params: QuadParams, derivatives: PositionDerivatives,
                        duration: float) -> LinearMPC:
        model = HoverModel.build(params, self.wind_mean, self.yaw, self.sample_time)
        reference = LinearizedReference(model, derivatives)
        return LinearMPC(model, self.settings, reference, duration)


def load_hover_mpc(path: str | Path) -> HoverMPCConfig:
    path = Path(path).resolve()
    section = load_toml(path).get("hover_mpc")
    if section is None:
        raise ValueError(f"{path} has no [hover_mpc] section")
    max_rate = float(section["max_body_rate"])
    # The LQR bounds ‖ω‖ ≤ max_body_rate (a ball); a QP needs linear bounds, so
    # use the largest box inside the ball: |ω_i| ≤ max_body_rate / √3.
    rate_box = max_rate / np.sqrt(3.0)
    thrust = _vector(section, "thrust_limits", 2)
    settings = LinearMPCSettings(
        Q=np.diag(_vector(section, "Q_diag", 9)),
        R=np.diag(_vector(section, "R_diag", 4)),
        horizon=int(section["horizon"]),
        terminal=str(section["terminal"]),
        u_lb=np.array([-rate_box] * 3 + [thrust[0]]),
        u_ub=np.array([rate_box] * 3 + [thrust[1]]),
        max_tilt=float(section["max_tilt"]),
    )
    return HoverMPCConfig(sample_time=float(section["sample_time"]),
                          yaw=np.radians(float(section["yaw_deg"])),
                          wind_mean=_vector(section, "wind_mean", 3),
                          settings=settings, config_path=path)


def position_derivatives(reference) -> PositionDerivatives:
    """(p, v, a, j)(t) of a datagen reference; a setpoint has zero derivatives."""
    if hasattr(reference, "derivatives"):
        return reference.derivatives
    if isinstance(reference, SetpointReference):
        p = reference(0.0)[0][:3]
        zero = np.zeros(3)
        return lambda t: (p, zero, zero, zero)
    raise ValueError(f"no position derivatives for reference {type(reference).__name__}")
