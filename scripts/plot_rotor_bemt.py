"""Plot how far BEMT thrust and torque depart from the quadratic model with airflow.

    PYTHONPATH=src python scripts/plot_rotor_bemt.py configs/rotor.toml --no-show

At the still-air hover rotor speed (mixer at f_c = m g, from configs/quadrotor.toml),
sweeps edgewise airspeed and axial (climb/descent) airspeed and plots
T_BEMT / T_quadratic and Q_BEMT / Q_quadratic. Writes figures/rotor_bemt.png in a run
directory; the ratios are 1 by construction in still air.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plotting import save_figures
from dynamics.params import load_params
from dynamics.rotor import bemt_thrust, load_rotor_params, mixer, quadratic_thrust_torque
from experiment.run_record import start_run

REPO_ROOT = Path(__file__).resolve().parents[1]


def ratios(omega, airspeeds, rotor) -> tuple[np.ndarray, np.ndarray]:
    """T and Q of BEMT over the quadratic model for each body-frame airspeed (k, 3)."""
    T_q, Q_q = quadratic_thrust_torque(omega, rotor)
    out = np.array([bemt_thrust(omega, np.tile(v, (4, 1)), rotor)[:2] for v in airspeeds])
    return out[:, 0, 0] / T_q[0], out[:, 1, 0] / Q_q[0]


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rotor_config", type=Path)
    parser.add_argument("--vehicle", type=Path, default=REPO_ROOT / "configs" / "quadrotor.toml")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    parser.add_argument("--no-show", dest="show", action="store_false")
    args = parser.parse_args(argv)

    rotor = load_rotor_params(args.rotor_config)
    vehicle = load_params(args.vehicle)
    run_dir = start_run(args.results_dir, "rotor_bemt", 0,
                        [args.rotor_config.resolve(), args.vehicle.resolve()], REPO_ROOT)
    omega = mixer(vehicle.mass * vehicle.gravity, np.zeros(3), rotor)
    v_h = rotor.hover_inflow * omega[0] * rotor.radius            # hover induced velocity, m/s

    edge = np.linspace(0.0, 8.0, 81)
    axial = np.linspace(-0.45 * v_h, 3.0, 81)                     # stop short of vortex ring
    T_e, Q_e = ratios(omega, np.column_stack([edge, 0 * edge, 0 * edge]), rotor)
    T_a, Q_a = ratios(omega, np.column_stack([0 * axial, 0 * axial, -axial]), rotor)  # FRD: up = −z

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, x, T, Q, label in ((axes[0], edge, T_e, Q_e, "edgewise airspeed [m/s]"),
                               (axes[1], axial, T_a, Q_a, "climb rate through the air [m/s]")):
        ax.plot(x, T, label="thrust T_BEMT / k_T Ω²")
        ax.plot(x, Q, "--", label="torque Q_BEMT / k_Q Ω²")
        ax.axhline(1.0, color="k", linewidth=0.6)
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("BEMT / quadratic model")
    axes[0].legend()
    axes[1].axvline(0.0, color="k", linewidth=0.6)
    fig.suptitle(f"Uniform-inflow BEMT at hover speed Ω = {omega[0]:.0f} rad/s "
                 f"({omega[0] * 60 / (2 * np.pi):.0f} rpm), v_h = {v_h:.2f} m/s")
    fig.tight_layout()
    save_figures({"rotor_bemt": fig}, run_dir / "figures")
    if args.show:
        plt.show()
    plt.close("all")
    return run_dir


if __name__ == "__main__":
    main()
