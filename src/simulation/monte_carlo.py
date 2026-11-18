"""Paired Monte Carlo runs over different wind realizations (Fig. 4, Tables II-III).

Paired design: every controller is run once per seed, and each of those runs
gets a freshly built episode (wind, LQR, rate controller) from that seed. So
for one seed all controllers fly the same wind, and differences between them
are not wind luck. This module knows nothing about configs: the caller
supplies a factory that builds an episode from a seed.
"""
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from controllers.low_level import BodyRateController
from controllers.lqr import LQRTracker
from dynamics.actuator import RotorActuator
from dynamics.params import QuadParams

from .closed_loop import Augmentation, Reference, SimulationLog, Timing, Wind, simulate


class Episode(Protocol):
    """Everything simulate() needs except the augmentation; fresh per run."""

    x0: np.ndarray
    duration: float
    reference: Reference
    wind: Wind
    params: QuadParams
    lqr: LQRTracker
    rate_controller: BodyRateController
    timing: Timing
    actuator: RotorActuator


@dataclass(frozen=True)
class RunResult:
    """One controller on one seed. augmentation is the object used, or None."""

    seed: int
    log: SimulationLog
    augmentation: Augmentation | None


def run_paired(
    make_episode: Callable[[int], Episode],
    controllers: Mapping[str, Callable[[], Augmentation | None]],
    seeds: Sequence[int],
) -> dict[str, list[RunResult]]:
    """Run every controller on every seed.

    Args:
        make_episode: seed -> a new Episode; must build new stateful objects
            (wind, LQR) on every call, so runs do not share state.
        controllers: name -> factory of a new augmentation, or of None for
            the LQR alone.
        seeds: One wind realization per seed.

    Returns:
        name -> results in seed order.
    """
    if not controllers:
        raise ValueError("no controllers to run")
    if not seeds:
        raise ValueError("no seeds to run")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct, or runs are not independent")
    results: dict[str, list[RunResult]] = {name: [] for name in controllers}
    for seed in seeds:
        for name, make_augmentation in controllers.items():
            episode = make_episode(seed)
            augmentation = make_augmentation()
            log = simulate(episode.x0, episode.duration, episode.reference, episode.wind,
                           episode.params, episode.lqr, episode.rate_controller,
                           episode.timing, augmentation=augmentation,
                           actuator=episode.actuator)
            results[name].append(RunResult(seed=seed, log=log, augmentation=augmentation))
    return results
