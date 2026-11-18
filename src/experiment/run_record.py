"""Per-run records: results/runs/<UTC timestamp>_<name>/ with everything needed to reproduce.

    metadata.json    git commit, dirty flag, untracked files, seed, UTC time,
                     package versions, command line
    config/*.toml    verbatim copies of every config the run used
    git_diff.patch   `git diff HEAD`, only when tracked files were modified
    log.npz          SimulationLog arrays and timing
"""
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import scipy

from simulation.closed_loop import SimulationLog, Timing

_LOG_FILE = "log.npz"
_TIMING_PREFIX = "timing_"


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=True
    )
    return result.stdout


def git_state(repo_root: str | Path) -> dict:
    """Commit hash, dirty flag, untracked files and diff of the working tree.

    Raises:
        RuntimeError: If repo_root is not inside a git repository; a run
            without a commit cannot be traced back to code.
    """
    try:
        commit = _git(Path(repo_root), "rev-parse", "HEAD").strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        raise RuntimeError(f"cannot read git commit in {repo_root}") from err
    status = _git(Path(repo_root), "status", "--porcelain").splitlines()
    untracked = [line[3:] for line in status if line.startswith("??")]
    diff = _git(Path(repo_root), "diff", "HEAD")
    return {
        "commit": commit,
        "dirty": bool(status),
        "untracked": untracked,
        "diff": diff,
    }


def start_run(results_root: str | Path, name: str, seed: int,
              config_paths: list[Path], repo_root: str | Path,
              now: datetime | None = None) -> Path:
    """Create the run directory and write metadata, configs and any git diff.

    Call before simulating, so a crashed run still leaves its provenance.

    Returns:
        The new run directory.
    """
    now = now or datetime.now(timezone.utc)
    run_dir = Path(results_root) / f"{now.strftime('%Y%m%dT%H%M%SZ')}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    config_dir = run_dir / "config"
    config_dir.mkdir()
    for path in config_paths:
        target = config_dir / Path(path).name
        if target.exists():
            raise ValueError(f"two configs share the file name {Path(path).name}")
        shutil.copyfile(path, target)

    git = git_state(repo_root)
    if git["diff"]:
        (run_dir / "git_diff.patch").write_text(git["diff"])

    metadata = {
        "name": name,
        "seed": seed,
        "utc_time": now.isoformat(),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "git_untracked": git["untracked"],
        "configs": [str(path) for path in config_paths],
        "command": sys.argv,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return run_dir


def save_log(run_dir: str | Path, log: SimulationLog) -> Path:
    """Write the log arrays and timing to run_dir/log.npz."""
    arrays = {f.name: getattr(log, f.name) for f in fields(log) if f.name != "timing"}
    timing = {_TIMING_PREFIX + key: value for key, value in asdict(log.timing).items()}
    path = Path(run_dir) / _LOG_FILE
    np.savez(path, **arrays, **timing)
    return path


def load_log(run_dir: str | Path) -> SimulationLog:
    """Inverse of save_log."""
    with np.load(Path(run_dir) / _LOG_FILE) as data:
        timing = Timing(**{
            f.name: float(data[_TIMING_PREFIX + f.name]) for f in fields(Timing)
        })
        arrays = {f.name: data[f.name] for f in fields(SimulationLog) if f.name != "timing"}
    return SimulationLog(**arrays, timing=timing)
