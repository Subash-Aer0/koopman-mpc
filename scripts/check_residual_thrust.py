"""Go/no-go check for the CCTA paper's Model 2: what does the residual thrust f_res measure?

    PYTHONPATH=src python scripts/check_residual_thrust.py \
        configs/experiments/figure8_gaussian.toml configs/experiments/circle_gaussian.toml \
        --vehicle configs/quadrotor_motor.toml --seeds 5

Flies the LQR (the controller under Model 1/2's data) on each experiment with the
given vehicle, and at every logged instant compares, mass-normalised (m/s², the
units of the thrust channel c):

    c_res  = (f_c − Σ k_T Ω_R²) / m     Model 2's residual: commanded minus thrust
                                         estimated from the measured rotor speeds
    c_miss = (f_c − Σ T_i) / m          thrust actually missing: commanded minus
                                         applied (BEMT at Ω_R, the simulator's truth)

f_c = m c is the commanded collective thrust. A perfect residual has c_res = c_miss.
Reported: RMS of both, their correlation, the fraction of samples where their signs
agree, the least-squares gain c_miss ≈ g c_res, and the rotor-speed droop Ω_R/Ω_cmd − 1.

Prediction recorded before running (plan of 2026-11-19): with the uniform-inflow
BEMT, edgewise flow raises thrust far more than torque, so the motor slows slightly
(c_res > 0, "thrust missing") while the rotor in fact delivers extra thrust
(c_miss < 0): c_res should be small and of the opposite sign to c_miss.

Writes summary.json in a run directory. The LQR runs without the paper's dither u_d,
which changes the command but not the relation between c_res and c_miss.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

from dynamics.quaternion import Quaternion
from dynamics.rotor import bemt_thrust, mixer, rotor_airspeeds
from experiment.builders import load_experiment
from experiment.run_record import start_run
from simulation.closed_loop import simulate

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("check_residual_thrust")


def residual_signals(log, exp) -> dict[str, np.ndarray]:
    """c_res, c_miss (T,) m/s² and speed droop (T, 4) at the logged instants of one flight."""
    act, rotor, mass = exp.actuator, exp.actuator.rotor, exp.params.mass
    c_res, c_miss, droop = [], [], []
    for x, u, w, omega_r in zip(log.x, log.u_cmd, log.v_w, log.omega_rotor):
        f_t, tau_t = exp.rate_controller(x, u)
        f_c = -f_t[2]
        omega_cmd = mixer(f_c, tau_t, rotor)
        R = Quaternion(x[3:7]).normalize().to_rotation_matrix()
        T, _, _ = bemt_thrust(omega_r, rotor_airspeeds(R.T @ (x[7:10] - w), x[10:13], rotor),
                              rotor)
        c_res.append((f_c - rotor.thrust_coefficient * np.sum(omega_r**2)) / mass)
        c_miss.append((f_c - np.sum(T)) / mass)
        droop.append(omega_r / omega_cmd - 1.0)
    return {"c_res": np.array(c_res), "c_miss": np.array(c_miss), "droop": np.array(droop)}


def statistics(c_res: np.ndarray, c_miss: np.ndarray, droop: np.ndarray) -> dict:
    nonzero = (np.abs(c_res) > 1e-9) & (np.abs(c_miss) > 1e-9)
    return {
        "samples": int(len(c_res)),
        "rms_c_res_m_s2": float(np.sqrt(np.mean(c_res**2))),
        "rms_c_miss_m_s2": float(np.sqrt(np.mean(c_miss**2))),
        "mean_c_res_m_s2": float(c_res.mean()),
        "mean_c_miss_m_s2": float(c_miss.mean()),
        "correlation": float(np.corrcoef(c_res, c_miss)[0, 1]) if np.std(c_res) > 0 else None,
        "sign_agreement": float(np.mean(np.sign(c_res[nonzero]) == np.sign(c_miss[nonzero])))
        if nonzero.any() else None,
        "lsq_gain_c_miss_on_c_res": float(c_res @ c_miss / (c_res @ c_res))
        if np.any(c_res) else None,
        "droop_pct": {"mean": float(100 * droop.mean()), "min": float(100 * droop.min()),
                      "max": float(100 * droop.max())},
    }


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiments", type=Path, nargs="+")
    parser.add_argument("--vehicle", type=Path, default=REPO_ROOT / "configs/quadrotor_motor.toml")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    args = parser.parse_args(argv)

    probe = load_experiment(args.experiments[0], vehicle_config=args.vehicle)
    if probe.actuator.rotor is None:
        raise ValueError(f"{args.vehicle} has no rotor model; f_res needs rotor speeds")
    config_paths = list(dict.fromkeys(
        p for e in args.experiments
        for p in load_experiment(e, vehicle_config=args.vehicle).config_paths))
    run_dir = start_run(args.results_dir, "residual_thrust_check", 0, config_paths, REPO_ROOT)

    summary = {"vehicle": str(args.vehicle.resolve()),
               "motor": None if probe.actuator.motor is None else vars(probe.actuator.motor),
               "seeds": list(range(args.seeds)), "experiments": {}}
    pooled = {"c_res": [], "c_miss": [], "droop": []}
    for path in args.experiments:
        signals = {"c_res": [], "c_miss": [], "droop": []}
        for seed in range(args.seeds):
            exp = load_experiment(path, seed=seed, vehicle_config=args.vehicle)
            log = simulate(exp.x0, exp.duration, exp.reference, exp.wind, exp.params, exp.lqr,
                           exp.rate_controller, timing=exp.timing, actuator=exp.actuator)
            for key, value in residual_signals(log, exp).items():
                signals[key].append(value)
                pooled[key].append(value)
        stats = statistics(*(np.concatenate(signals[k]) for k in ("c_res", "c_miss", "droop")))
        summary["experiments"][exp.name] = stats
        logger.info("%s: %s", exp.name, json.dumps(stats))
    summary["pooled"] = statistics(*(np.concatenate(pooled[k]) for k in ("c_res", "c_miss", "droop")))
    logger.info("pooled: %s", json.dumps(summary["pooled"]))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("results in %s", run_dir)
    return run_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    main()
