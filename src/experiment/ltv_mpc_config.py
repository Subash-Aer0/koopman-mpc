"""Load the LTV-MPC config (configs/controllers/ltv_mpc.toml) and build the model and MPC.

The LTV config names a hover-MPC base config and adds only the terminal mode
and the Riccati lead, so weights, limits, horizon, sample time, mean wind and
yaw are shared with the hover MPC by construction (a fair comparison).

The model depends only on the reference, vehicle and mean wind, not on the
wind seed, so it is built once per experiment (make_model, seconds) and a
fresh controller is made from it per run (make_controller, milliseconds).

What the controller believes about the vehicle is set by `model_drag_scale`:
its model uses drag_coeffs × scale, while the simulator keeps the true values.
    1.0   oracle: the simulator's own drag model (an upper bound, simulation only)
    0.0   nominal: rigid body only, no knowledge of drag or of how wind acts
A grey-box controller instead believes drag coefficients fitted from flight
data (with_drag_coeffs), which replace the scaled true values.
Mass, inertia and gravity are taken as known in every case.
"""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from controllers.hover_model import HoverModel
from controllers.linear_mpc import LinearMPCSettings, PositionDerivatives
from controllers.ltv_mpc import LTVMPC, TERMINAL_MODES
from controllers.trajectory_model import LTVModel
from dynamics.params import QuadParams

from .builders import load_toml
from .hover_mpc_config import HoverMPCConfig, load_hover_mpc


@dataclass(frozen=True)
class LTVMPCConfig:
    """A hover-MPC base config plus the LTV-specific settings, and the config paths."""

    base: HoverMPCConfig
    terminal: str
    riccati_lead: float
    model_drag_scale: float      # controller's drag = true drag × this, dimensionless
    config_path: Path
    # If set, the controller's drag coefficients (3,), N·s²/m², instead of the scaled true ones.
    model_drag_coeffs: np.ndarray | None = None

    @property
    def settings(self) -> LinearMPCSettings:
        """The base settings with this config's terminal mode."""
        return replace(self.base.settings, terminal=self.terminal)

    @property
    def config_paths(self) -> list[Path]:
        """Both files, for the run record."""
        return [self.config_path, self.base.config_path]

    def with_drag_coeffs(self, drag_coeffs) -> "LTVMPCConfig":
        """The same config believing the given drag coefficients (e.g. a grey-box fit)."""
        drag = np.asarray(drag_coeffs, dtype=float)
        if drag.shape != (3,) or not np.all(np.isfinite(drag)) or np.any(drag < 0):
            raise ValueError(f"drag_coeffs must be 3 finite non-negative values, got {drag}")
        return replace(self, model_drag_coeffs=drag)

    def model_params(self, params: QuadParams) -> QuadParams:
        """The vehicle as the controller believes it: `params` with its drag replaced."""
        if self.model_drag_coeffs is not None:
            return replace(params, drag_coeffs=self.model_drag_coeffs)
        return replace(params, drag_coeffs=params.drag_coeffs * self.model_drag_scale)

    def hover_model(self, params: QuadParams) -> HoverModel:
        """Hover linearization of the believed vehicle, at the base config's trim."""
        base = self.base
        return HoverModel.build(self.model_params(params), base.wind_mean, base.yaw,
                                base.sample_time)

    def make_model(self, params: QuadParams, derivatives: PositionDerivatives,
                   duration: float) -> LTVModel:
        """Model along the reference for a run of `duration` s (plus one horizon and the lead).

        Args:
            params: The true vehicle; the model uses model_params(params).
        """
        base = self.base
        span = duration + base.settings.horizon * base.sample_time + self.riccati_lead
        return LTVModel.build(derivatives, span, self.model_params(params), base.wind_mean,
                              base.yaw, base.sample_time)

    def make_controller(self, model: LTVModel) -> LTVMPC:
        return LTVMPC(model, self.settings, self.riccati_lead)


def load_ltv_mpc(path: str | Path) -> LTVMPCConfig:
    path = Path(path).resolve()
    section = load_toml(path).get("ltv_mpc")
    if section is None:
        raise ValueError(f"{path} has no [ltv_mpc] section")
    for key in ("base_config", "terminal", "riccati_lead", "model_drag_scale"):
        if key not in section:
            raise ValueError(f"[ltv_mpc] missing '{key}'")
    terminal = str(section["terminal"])
    if terminal not in TERMINAL_MODES:
        raise ValueError(f"[ltv_mpc] terminal must be one of {TERMINAL_MODES}, got {terminal!r}")
    lead = float(section["riccati_lead"])
    if not lead >= 0:   # also rejects NaN
        raise ValueError(f"[ltv_mpc] riccati_lead must be non-negative, got {lead}")
    drag_scale = float(section["model_drag_scale"])
    if not (np.isfinite(drag_scale) and drag_scale >= 0):
        raise ValueError(f"[ltv_mpc] model_drag_scale must be finite and non-negative, "
                         f"got {drag_scale}")
    base = load_hover_mpc(path.parent / section["base_config"])
    return LTVMPCConfig(base=base, terminal=terminal, riccati_lead=lead,
                        model_drag_scale=drag_scale, config_path=path)
