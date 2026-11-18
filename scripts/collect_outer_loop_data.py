"""Collect dithered LTV-MPC flights for the outer-loop Koopman model.

    PYTHONPATH=src python scripts/collect_outer_loop_data.py configs/datagen/outer_loop.toml

The flights are flown by the LTV MPC named in [datagen] mpc_config, normally
the nominal one (configs/controllers/ltv_mpc_nominal.toml): the controller
available before learning, which does not know how the wind acts.

Episode kinds per split: random sum-of-sines references ([random_reference]),
circles (optional), and hover from a random offset.

Writes results/runs/<time>_outer_loop_data/ with the run record, the dataset
(datagen.outer_loop.save_episodes) and episodes.json (one line per episode).
Each episode gets its own wind and dither streams, drawn from the config seed.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

from datagen.outer_loop import DitheredController, episode_from_log, save_episodes
from datagen.trajectories import CircleReference, SetpointReference, sample_sum_of_sines
from dynamics import dynamics as rigid_body
from dynamics.actuator import load_actuator
from dynamics.params import load_params
from dynamics.quaternion import Quaternion
from experiment.builders import build_rate_controller, build_wind, load_toml, state_on_reference
from experiment.hover_mpc_config import position_derivatives
from experiment.ltv_mpc_config import load_ltv_mpc
from experiment.run_record import start_run
from simulation.outer_loop import simulate_outer_loop

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("collect_outer_loop_data")


def episode_specs(split_cfg: dict, common: dict, random_cfg: dict | None, gravity: float,
                  rng) -> list[tuple]:
    """(name, reference, x0) per episode of one split.

    Keys of split_cfg: random_episodes, circle_periods, circle_radii (each
    optional, default none) and hover_episodes. random_cfg is the
    [random_reference] table, required when random_episodes > 0.
    """
    center = np.asarray(common["center"], dtype=float)
    yaw = np.radians(float(common["yaw_deg"]))
    specs = []
    n_random = int(split_cfg.get("random_episodes", 0))
    if n_random > 0 and random_cfg is None:
        raise ValueError("random_episodes > 0 needs a [random_reference] table")
    for i in range(n_random):
        tilt_lo, tilt_hi = random_cfg["peak_tilt_range"]
        height_lo, height_hi = random_cfg["vertical_amplitude_range"]
        ref = sample_sum_of_sines(
            rng, center, int(random_cfg["components"]), random_cfg["frequency_range"],
            peak_tilt=rng.uniform(tilt_lo, tilt_hi), max_speed=float(random_cfg["max_speed"]),
            vertical_amplitude=rng.uniform(height_lo, height_hi), yaw=yaw, gravity=gravity)
        specs.append((f"random_{i}", ref, state_on_reference(ref)))
    for period in split_cfg.get("circle_periods", []):
        for radius in split_cfg.get("circle_radii", []):
            ref = CircleReference(center, radius, 0.0, period, yaw, gravity)
            specs.append((f"circle_T{period:g}_r{radius:g}", ref, state_on_reference(ref)))
    offset = np.asarray(common["hover_offset"], dtype=float)
    for i in range(int(split_cfg["hover_episodes"])):
        ref = SetpointReference(center, yaw, gravity)
        x0 = np.zeros(rigid_body.STATE_DIM)
        x0[rigid_body.POS] = center + rng.uniform(-offset, offset)
        x0[rigid_body.QUAT] = Quaternion.from_euler([0.0, 0.0, yaw]).to_numpy()
        specs.append((f"hover_{i}", ref, x0))
    return specs


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    args = parser.parse_args(argv)

    path = args.config.resolve()
    cfg = load_toml(path)
    section = cfg["datagen"]
    vehicle_path = (path.parent / section["vehicle_config"]).resolve()
    controller_path = (path.parent / section["controller_config"]).resolve()
    mpc_path = (path.parent / section["mpc_config"]).resolve()
    params = load_params(vehicle_path)
    actuator = load_actuator(load_toml(vehicle_path), vehicle_path)   # the true rotors
    controller_cfg = load_toml(controller_path)
    dt_sim = float(controller_cfg["timing"]["dt_sim"])
    mpc_cfg = load_ltv_mpc(mpc_path)
    sample_time = mpc_cfg.base.sample_time
    duration = float(section["duration"])
    dither = np.asarray(cfg["dither"]["bounds"], dtype=float)
    seed = int(section["seed"])

    run_dir = start_run(args.results_dir, section["name"], seed,
                        [path, vehicle_path, controller_path, *actuator.config_paths,
                         *mpc_cfg.config_paths], REPO_ROOT)
    seeds = np.random.SeedSequence(seed)
    spec_rng = np.random.default_rng(seeds.spawn(1)[0])
    splits, listing = {}, []
    for split in ("train", "validation"):
        specs = episode_specs(cfg["splits"][split], cfg["common"], cfg.get("random_reference"),
                              params.gravity, spec_rng)
        episodes = []
        for (name, ref, x0), child in zip(specs, seeds.spawn(len(specs))):
            wind_rng, dither_rng = (np.random.default_rng(s) for s in child.spawn(2))
            mpc = mpc_cfg.make_controller(
                mpc_cfg.make_model(params, position_derivatives(ref), duration))
            controller = DitheredController(mpc, dither, mpc_cfg.settings.u_lb,
                                            mpc_cfg.settings.u_ub, dither_rng)
            log = simulate_outer_loop(x0, duration, ref, build_wind(cfg, wind_rng), params,
                                      controller, build_rate_controller(controller_cfg, params),
                                      dt_sim=dt_sim, dt_log=sample_time, actuator=actuator)
            episodes.append(episode_from_log(name, log))
            ep = episodes[-1]
            max_tilt = float(np.max(np.abs(ep.x9[:, 6:8])))
            max_speed = float(np.max(np.linalg.norm(ep.x9[:, 3:6], axis=1)))
            listing.append({"split": split, "name": name, "samples": len(ep.u),
                            "max_tilt_rad": max_tilt, "max_speed_m_s": max_speed,
                            "fallbacks": mpc.fallback_count})
            logger.info("%-10s %-16s max tilt %.3f rad, max speed %.2f m/s, fallbacks %d",
                        split, name, max_tilt, max_speed, mpc.fallback_count)
        splits[split] = episodes
    save_episodes(run_dir, splits)
    (run_dir / "episodes.json").write_text(json.dumps(listing, indent=2))
    logger.info("dataset written to %s", run_dir)
    return run_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    main()
