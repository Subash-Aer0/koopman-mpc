"""Generate the identification dataset (train + validation splits) from a datagen config.

    PYTHONPATH=src python scripts/01_generate_data.py configs/datagen/model1.toml

Writes results/runs/<time>_<name>/ with metadata, config copies and dataset.npz
(see datagen/dataset.py). Each episode gets its own child seed, spawned from
[datagen] seed in episode order, with separate streams for wind and
perturbations: any single episode can be regenerated without the others.
"""
import argparse
import logging
from pathlib import Path

import numpy as np

from datagen.collect import HeldPerturbations, Split, random_perturbations, split_from_logs
from datagen.dataset import save_dataset
from experiment.builders import DatagenConfig, load_datagen, state_on_reference
from experiment.run_record import start_run
from simulation.closed_loop import SimulationLog, simulate

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("generate_data")


def run_episode(cfg: DatagenConfig, reference_name: str,
                seed: np.random.SeedSequence) -> SimulationLog:
    """One closed-loop episode with fresh controllers, wind and perturbations."""
    wind_seed, perturbation_seed = seed.spawn(2)
    n_augment = round(cfg.duration / cfg.timing.dt_augment)
    u_d = random_perturbations(n_augment, cfg.perturbation_bounds,
                               np.random.default_rng(perturbation_seed))
    reference = cfg.references[reference_name]
    lqr, rate_controller = cfg.make_controllers()
    return simulate(
        state_on_reference(reference), cfg.duration, reference,
        cfg.make_wind(np.random.default_rng(wind_seed)), cfg.params,
        lqr, rate_controller, cfg.timing,
        augmentation=HeldPerturbations(u_d, cfg.timing.dt_augment),
        actuator=cfg.actuator,
    )


def generate(cfg: DatagenConfig) -> dict[str, Split]:
    """Simulate every episode in cfg.episodes and group them into splits."""
    seeds = np.random.SeedSequence(cfg.seed).spawn(len(cfg.episodes))
    logs: dict[str, list] = {}
    for i, (episode, seed) in enumerate(zip(cfg.episodes, seeds)):
        logger.info("episode %d/%d: %s, %s #%d", i + 1, len(cfg.episodes),
                    episode.split, episode.reference, episode.repeat)
        logs.setdefault(episode.split, []).append((run_episode(cfg, episode.reference, seed),
                                                   episode))
    return {
        split: split_from_logs([log for log, _ in entries],
                               [episode.reference for _, episode in entries],
                               [episode.repeat for _, episode in entries])
        for split, entries in logs.items()
    }


def summarize(name: str, split: Split) -> None:
    """Log sanity statistics: pair count, tracking error, wind moments, saturation."""
    rms = np.sqrt(np.mean(split.e_x**2, axis=(0, 1)))
    wind = split.v_w.reshape(-1, 3)
    saturated = np.any(~np.isclose(split.u_cmd, split.u_h + split.u_d), axis=-1)
    logger.info("%s: %d episodes, %d snapshot pairs", name, len(split.reference), split.n_pairs)
    logger.info("%s: RMS e_x pos %s m, vel %s m/s, rpy %s rad", name,
                np.round(rms[0:3], 3), np.round(rms[3:6], 3), np.round(rms[6:9], 3))
    logger.info("%s: wind mean %s m/s, var %s (m/s)^2", name,
                np.round(wind.mean(axis=0), 3), np.round(wind.var(axis=0), 3))
    logger.info("%s: %.1f%% of samples have u_h + u_d saturated", name, 100 * saturated.mean())


def main(argv: list[str] | None = None) -> Path:
    """Generate the dataset, write results/runs/<time>_<name>/, return that directory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="datagen TOML file")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    args = parser.parse_args(argv)

    cfg = load_datagen(args.config)
    run_dir = start_run(args.results_dir, cfg.name, cfg.seed, cfg.config_paths, REPO_ROOT)
    logger.info("run %s: seed %d, %d episodes of %.1f s", run_dir.name, cfg.seed,
                len(cfg.episodes), cfg.duration)

    splits = generate(cfg)
    for name, split in splits.items():
        summarize(name, split)
    path = save_dataset(run_dir, splits)
    logger.info("saved %s", path)
    return run_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    main()
