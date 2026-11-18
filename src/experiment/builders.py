"""Build simulation objects from TOML configs.

The only module that knows the config layout (see configs/baseline.toml,
configs/experiments/*.toml, configs/datagen/*.toml and configs/controllers/*.toml);
everything it builds takes plain arrays, so the algorithm code stays independent of files.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import tomllib

import numpy as np

from controllers import sdlqr_model as lqr_model
from controllers.kmpc import KMPCSettings, KoopmanMPC
from controllers.low_level import BodyRateController
from controllers.lqr import LQRTracker
from datagen.trajectories import CircleReference, Figure8Reference, SetpointReference
from dynamics import dynamics as rigid_body
from dynamics.actuator import RotorActuator, load_actuator
from dynamics.params import QuadParams, load_params
from dynamics.quaternion import Quaternion
from koopman.lifting import MonomialLifting
from koopman.models import KoopmanModel
from koopman.preprocessing import Standardizer
from simulation.closed_loop import Reference, Timing, Wind
from wind.wind_model import ConstantWind, GaussianWind


def load_toml(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _section(cfg: dict, name: str) -> dict:
    if name not in cfg:
        raise ValueError(f"config is missing section [{name}]")
    return cfg[name]


def _value(section: dict, key: str, where: str):
    if key not in section:
        raise ValueError(f"[{where}] is missing key '{key}'")
    return section[key]


def _vector(section: dict, key: str, size: int, where: str) -> np.ndarray:
    value = np.asarray(_value(section, key, where), dtype=float)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f"[{where}] {key} must be {size} finite numbers, got {value.tolist()}")
    return value


def build_timing(controller_cfg: dict) -> Timing:
    section = _section(controller_cfg, "timing")
    keys = ("dt_sim", "dt_control", "dt_gain", "dt_augment", "dt_log")
    return Timing(**{key: float(_value(section, key, "timing")) for key in keys})


def build_lqr(controller_cfg: dict, timing: Timing) -> LQRTracker:
    """LQRTracker discretized with timing.dt_control, so the two cannot disagree."""
    section = _section(controller_cfg, "lqr")
    return LQRTracker(
        Q=np.diag(_vector(section, "Q_diag", 10, "lqr")),
        R=np.diag(_vector(section, "R_diag", 4, "lqr")),
        dt=timing.dt_control,
        max_body_rate=float(_value(section, "max_body_rate", "lqr")),
        thrust_limits=tuple(_vector(section, "thrust_limits", 2, "lqr")),
    )


def build_rate_controller(controller_cfg: dict, params: QuadParams) -> BodyRateController:
    section = _section(controller_cfg, "rate_controller")
    return BodyRateController(params, _vector(section, "gains", 3, "rate_controller"))


def state_on_reference(reference: Reference, t: float = 0.0) -> np.ndarray:
    """13-state [x_ref(t), ω_ref(t)]: exactly on the reference, so no start-up transient."""
    x_ref, u_ref = reference(t)
    x = np.empty(rigid_body.STATE_DIM)
    x[rigid_body.POS] = x_ref[lqr_model.POS]
    x[rigid_body.QUAT] = x_ref[lqr_model.QUAT]
    x[rigid_body.VEL] = x_ref[lqr_model.VEL]
    x[rigid_body.RATE] = u_ref[lqr_model.RATE]
    return x


def build_initial_state(experiment_cfg: dict, reference: Reference) -> np.ndarray:
    """13-state x0.

    With start_on_reference = true, x0 = state_on_reference(reference). Otherwise
    from position (m), Euler angles (deg, ZYX), velocity (m/s) and body rate (rad/s).
    """
    section = _section(experiment_cfg, "initial_state")
    where = "initial_state"
    if section.get("start_on_reference", False):
        return state_on_reference(reference)
    x0 = np.empty(rigid_body.STATE_DIM)
    x0[rigid_body.POS] = _vector(section, "position", 3, where)
    euler = np.radians(_vector(section, "euler_deg", 3, where))
    x0[rigid_body.QUAT] = Quaternion.from_euler(euler).to_numpy()
    x0[rigid_body.VEL] = _vector(section, "velocity", 3, where)
    x0[rigid_body.RATE] = _vector(section, "body_rate", 3, where)
    return x0


def build_reference(experiment_cfg: dict, gravity: float) -> Reference:
    return _reference_from_section(_section(experiment_cfg, "reference"), "reference", gravity)


def _reference_from_section(section: dict, where: str, gravity: float) -> Reference:
    kind = _value(section, "type", where)
    yaw = np.radians(float(_value(section, "yaw_deg", where)))
    if kind == "setpoint":
        return SetpointReference(_vector(section, "position", 3, where), yaw, gravity)
    if kind == "figure8":
        return Figure8Reference(
            center=_vector(section, "center", 3, where),
            amplitudes=_vector(section, "amplitudes", 3, where),
            period=float(_value(section, "period", where)),
            yaw=yaw,
            gravity=gravity,
        )
    if kind == "circle":
        return CircleReference(
            center=_vector(section, "center", 3, where),
            radius=float(_value(section, "radius", where)),
            vertical_amplitude=float(_value(section, "vertical_amplitude", where)),
            period=float(_value(section, "period", where)),
            yaw=yaw,
            gravity=gravity,
        )
    raise ValueError(f"[{where}] unknown reference type '{kind}'; "
                     "known: 'setpoint', 'figure8', 'circle'")


def build_wind(experiment_cfg: dict, rng: np.random.Generator) -> Wind:
    """Wind model; stochastic models draw from rng (unused by 'constant')."""
    section = _section(experiment_cfg, "wind")
    kind = _value(section, "type", "wind")
    if kind == "constant":
        return ConstantWind(_vector(section, "velocity", 3, "wind"))
    if kind == "gaussian":
        cov = np.asarray(_value(section, "cov", "wind"), dtype=float)
        if cov.shape != (3, 3):
            raise ValueError(f"[wind] cov must be a 3x3 matrix, got shape {cov.shape}")
        return GaussianWind(
            mean=_vector(section, "mean", 3, "wind"),
            cov=cov,
            hold_period=float(_value(section, "hold_period", "wind")),
            rng=rng,
        )
    raise ValueError(f"unknown wind type '{kind}'; known: 'constant', 'gaussian'")


@dataclass
class Experiment:
    """Everything simulate() needs, plus what a run record must store.

    Controllers are stateful (the LQR caches its gain): build a new
    Experiment for every run.
    """

    name: str
    seed: int
    duration: float
    x0: np.ndarray
    params: QuadParams
    lqr: LQRTracker
    rate_controller: BodyRateController
    timing: Timing
    reference: Reference
    wind: Wind
    rng: np.random.Generator
    config_paths: list[Path]
    actuator: RotorActuator      # the vehicle's rotors (truth); pass to simulate*


def _seed(section: dict, where: str) -> int:
    seed = _value(section, "seed", where)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"[{where}] seed must be a non-negative integer, got {seed!r}")
    return seed


def load_experiment(path: str | Path, seed: int | None = None) -> Experiment:
    """Load an experiment TOML and the vehicle/controller configs it points to.

    vehicle_config and controller_config are resolved relative to the
    experiment file. The single rng is seeded from [experiment] seed, or from
    `seed` when given (Monte Carlo runs: one experiment, many wind draws).
    """
    path = Path(path).resolve()
    cfg = load_toml(path)
    section = _section(cfg, "experiment")
    vehicle_path = (path.parent / _value(section, "vehicle_config", "experiment")).resolve()
    controller_path = (path.parent / _value(section, "controller_config", "experiment")).resolve()
    if seed is None:
        seed = _seed(section, "experiment")
    elif not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    seed = int(seed)

    params = load_params(vehicle_path)
    actuator = load_actuator(load_toml(vehicle_path), vehicle_path)
    controller_cfg = load_toml(controller_path)
    timing = build_timing(controller_cfg)
    rng = np.random.default_rng(seed)
    reference = build_reference(cfg, params.gravity)
    return Experiment(
        name=str(_value(section, "name", "experiment")),
        seed=seed,
        duration=float(_value(section, "duration", "experiment")),
        x0=build_initial_state(cfg, reference),
        params=params,
        lqr=build_lqr(controller_cfg, timing),
        rate_controller=build_rate_controller(controller_cfg, params),
        timing=timing,
        reference=reference,
        wind=build_wind(cfg, rng),
        rng=rng,
        config_paths=[path, vehicle_path, controller_path, *actuator.config_paths],
        actuator=actuator,
    )


@dataclass(frozen=True)
class EpisodeSpec:
    """One data-collection episode: which split, which reference, which repeat."""

    split: str
    reference: str
    repeat: int


@dataclass
class DatagenConfig:
    """A data-generation config (configs/datagen/*.toml), resolved to objects.

    References are stateless and shared by episodes; controllers and wind are
    stateful, so make_controllers / make_wind build fresh ones per episode.
    """

    name: str
    seed: int
    duration: float
    params: QuadParams
    timing: Timing
    references: dict[str, Reference]
    episodes: list[EpisodeSpec]
    perturbation_bounds: np.ndarray
    config_paths: list[Path]
    actuator: RotorActuator      # the vehicle's rotors (truth); pass to simulate
    _controller_cfg: dict
    _wind_cfg: dict

    def make_controllers(self) -> tuple[LQRTracker, BodyRateController]:
        return (build_lqr(self._controller_cfg, self.timing),
                build_rate_controller(self._controller_cfg, self.params))

    def make_wind(self, rng: np.random.Generator) -> Wind:
        return build_wind(self._wind_cfg, rng)


def _episodes(cfg: dict, references: dict[str, Reference]) -> list[EpisodeSpec]:
    """Expand [splits.<name>] references x repeats, reference-major, in file order."""
    splits = _section(cfg, "splits")
    if not splits:
        raise ValueError("[splits] must define at least one split")
    episodes = []
    for split, section in splits.items():
        where = f"splits.{split}"
        if not split.isidentifier():
            raise ValueError(f"[{where}] split name must be an identifier")
        names = _value(section, "references", where)
        repeats = _value(section, "repeats", where)
        if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
            raise ValueError(f"[{where}] repeats must be a positive integer, got {repeats!r}")
        if not names:
            raise ValueError(f"[{where}] references must not be empty")
        for name in names:
            if name not in references:
                raise ValueError(f"[{where}] unknown reference '{name}'; "
                                 f"defined: {sorted(references)}")
            episodes += [EpisodeSpec(split, name, r) for r in range(repeats)]
    return episodes


def load_datagen(path: str | Path) -> DatagenConfig:
    """Load a data-generation TOML and the vehicle/controller configs it points to.

    vehicle_config and controller_config are resolved relative to the file.
    """
    path = Path(path).resolve()
    cfg = load_toml(path)
    section = _section(cfg, "datagen")
    vehicle_path = (path.parent / _value(section, "vehicle_config", "datagen")).resolve()
    controller_path = (path.parent / _value(section, "controller_config", "datagen")).resolve()

    params = load_params(vehicle_path)
    actuator = load_actuator(load_toml(vehicle_path), vehicle_path)
    controller_cfg = load_toml(controller_path)
    timing = build_timing(controller_cfg)
    duration = float(_value(section, "duration", "datagen"))
    n_augment = duration / timing.dt_augment
    if not (duration > 0 and abs(n_augment - round(n_augment)) < 1e-9 * max(1.0, n_augment)):
        raise ValueError(f"[datagen] duration = {duration} s must be a positive multiple "
                         f"of dt_augment = {timing.dt_augment} s")

    reference_sections = _section(cfg, "references")
    references = {
        name: _reference_from_section(ref, f"references.{name}", params.gravity)
        for name, ref in reference_sections.items()
    }
    bounds = _vector(_section(cfg, "perturbation"), "bounds", lqr_model.INPUT_DIM, "perturbation")
    if np.any(bounds < 0):
        raise ValueError(f"[perturbation] bounds must be non-negative, got {bounds.tolist()}")
    wind_cfg = {"wind": _section(cfg, "wind")}
    build_wind(wind_cfg, np.random.default_rng(0))  # fail at load time, not mid-dataset

    return DatagenConfig(
        name=str(_value(section, "name", "datagen")),
        seed=_seed(section, "datagen"),
        duration=duration,
        params=params,
        timing=timing,
        references=references,
        episodes=_episodes(cfg, references),
        perturbation_bounds=bounds,
        config_paths=[path, vehicle_path, controller_path, *actuator.config_paths],
        actuator=actuator,
        _controller_cfg=controller_cfg,
        _wind_cfg=wind_cfg,
    )


@dataclass(frozen=True)
class IdentifyConfig:
    """An identification config (configs/identify/*.toml)."""

    name: str
    seed: int
    degree: int
    l1_lambda: float
    horizon: int
    config_paths: list[Path]


def load_identify(path: str | Path) -> IdentifyConfig:
    path = Path(path).resolve()
    section = _section(load_toml(path), "identify")
    degree = _value(section, "degree", "identify")
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
        raise ValueError(f"[identify] degree must be a positive integer, got {degree!r}")
    l1_lambda = float(_value(section, "l1_lambda", "identify"))
    if not (np.isfinite(l1_lambda) and l1_lambda >= 0):
        raise ValueError(f"[identify] l1_lambda must be non-negative and finite, got {l1_lambda}")
    horizon = _value(section, "horizon", "identify")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(f"[identify] horizon must be a positive integer, got {horizon!r}")
    return IdentifyConfig(
        name=str(_value(section, "name", "identify")),
        seed=_seed(section, "identify"),
        degree=degree,
        l1_lambda=l1_lambda,
        horizon=horizon,
        config_paths=[path],
    )


def _bound_vector(section: dict, key: str, size: int, where: str) -> np.ndarray:
    """Like _vector, but ±inf (an unbounded channel) is allowed."""
    value = np.asarray(_value(section, key, where), dtype=float)
    if value.shape != (size,) or np.any(np.isnan(value)):
        raise ValueError(f"[{where}] {key} must be {size} numbers or ±inf, got {value.tolist()}")
    return value


@dataclass
class KMPCConfig:
    """A K-MPC config (configs/controllers/kmpc.toml) and the model it points to.

    The model is loaded once; KoopmanMPC keeps a warm start, so
    make_controller builds a fresh one per run.
    """

    model_run: Path
    sample_time: float
    model: KoopmanModel
    lifting: MonomialLifting
    standardizer: Standardizer
    settings: KMPCSettings
    config_paths: list[Path]

    def make_controller(self) -> KoopmanMPC:
        return KoopmanMPC(self.model, self.lifting, self.standardizer, self.settings)

    def check_timing(self, timing: Timing) -> None:
        """Raise unless u_d is held for one model step, dt_augment == sample_time."""
        if not np.isclose(timing.dt_augment, self.sample_time):
            raise ValueError(f"dt_augment = {timing.dt_augment} s, but the model was "
                             f"identified at sample_time = {self.sample_time} s")


def load_kmpc(path: str | Path) -> KMPCConfig:
    """Load a K-MPC config and the identification run it names (model_run).

    model_run is resolved relative to the config file and must contain
    model.npz, standardizer.npz and identification.json (scripts/02_identify_models.py).
    """
    path = Path(path).resolve()
    section = _section(load_toml(path), "kmpc")
    model_run = (path.parent / _value(section, "model_run", "kmpc")).resolve()
    if not model_run.is_dir():
        raise ValueError(f"[kmpc] model_run {model_run} is not a directory")
    identification = json.loads((model_run / "identification.json").read_text())
    model = KoopmanModel.load(model_run / "model.npz")
    with np.load(model_run / "standardizer.npz") as scaler:
        standardizer = Standardizer(mean=scaler["mean"], std=scaler["std"])
    lifting = MonomialLifting(identification["state_dim"], identification["degree"])

    sample_time = float(_value(section, "sample_time", "kmpc"))
    if not (np.isfinite(sample_time) and sample_time > 0):
        raise ValueError(f"[kmpc] sample_time must be positive, got {sample_time}")
    horizon = _value(section, "horizon", "kmpc")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(f"[kmpc] horizon must be a positive integer, got {horizon!r}")
    n, m, d = model.C.shape[0], model.B.shape[1], model.D.shape[1]
    u_bound = _vector(section, "u_bound", m, "kmpc")
    if np.any(u_bound < 0):
        raise ValueError(f"[kmpc] u_bound must be non-negative, got {u_bound.tolist()}")
    settings = KMPCSettings(
        Q=np.diag(_vector(section, "Q_diag", n, "kmpc")),
        R=np.diag(_vector(section, "R_diag", m, "kmpc")),
        horizon=horizon,
        u_lb=-u_bound,
        u_ub=u_bound,
        h_lb=_bound_vector(section, "h_lb", n, "kmpc"),
        h_ub=_bound_vector(section, "h_ub", n, "kmpc"),
        wind_mean=_vector(section, "wind_mean", d, "kmpc"),
    )
    config = KMPCConfig(model_run=model_run, sample_time=sample_time, model=model,
                        lifting=lifting, standardizer=standardizer, settings=settings,
                        config_paths=[path])
    config.make_controller()  # fail here, not mid-run, on mismatched model files
    return config


@dataclass(frozen=True)
class EvaluationConfig:
    """A controller-comparison config (configs/evaluate/*.toml).

    Attributes:
        name: Run name.
        seed: Root seed; run_seeds() derives one wind seed per Monte Carlo run.
        n_runs: Monte Carlo runs per experiment and controller.
        experiments: Experiment TOMLs, each run with every controller.
        kmpc: K-MPC TOML.
        config_paths: The evaluation TOML (the others are added by the caller).
    """

    name: str
    seed: int
    n_runs: int
    experiments: list[Path]
    kmpc: Path
    config_paths: list[Path]

    def run_seeds(self) -> list[int]:
        """n_runs distinct seeds from SeedSequence(seed), independent of datagen's spawn tree."""
        seeds = np.random.SeedSequence(self.seed).generate_state(self.n_runs).tolist()
        if len(set(seeds)) != len(seeds):
            raise ValueError("derived run seeds collide; choose another root seed")
        return seeds


def load_evaluation(path: str | Path) -> EvaluationConfig:
    """Load an evaluation config; experiment and kmpc paths are relative to it."""
    path = Path(path).resolve()
    section = _section(load_toml(path), "evaluate")
    n_runs = _value(section, "n_runs", "evaluate")
    if not isinstance(n_runs, int) or isinstance(n_runs, bool) or n_runs < 2:
        raise ValueError(f"[evaluate] n_runs must be an integer ≥ 2 (a paired test needs "
                         f"two runs), got {n_runs!r}")
    experiments = _value(section, "experiments", "evaluate")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("[evaluate] experiments must be a non-empty list of paths")
    resolved = [(path.parent / p).resolve() for p in experiments]
    names = [load_toml(p)["experiment"]["name"] for p in resolved]
    if len(set(names)) != len(names):
        raise ValueError(f"[evaluate] experiment names must be distinct, got {names}")
    return EvaluationConfig(
        name=str(_value(section, "name", "evaluate")),
        seed=_seed(section, "evaluate"),
        n_runs=n_runs,
        experiments=resolved,
        kmpc=(path.parent / _value(section, "kmpc_config", "evaluate")).resolve(),
        config_paths=[path],
    )
