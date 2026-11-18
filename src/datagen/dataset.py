"""Save and load identification datasets: one dataset.npz of named Splits.

Keys are "<split>.<field>" (e.g. "train.e_x"); strings are stored as unicode
arrays, so loading never needs pickle. Provenance (configs, seed, commit) is
written next to the file by experiment.run_record.start_run.
"""
from pathlib import Path

import numpy as np

from datagen.collect import ARRAY_FIELDS, Split

DATASET_FILE = "dataset.npz"


def save_dataset(directory: str | Path, splits: dict[str, Split]) -> Path:
    """Write splits to directory/dataset.npz and return the path."""
    arrays = {}
    for name, split in splits.items():
        if not name.isidentifier():
            raise ValueError(f"split name must be an identifier, got {name!r}")
        arrays.update({f"{name}.{field}": getattr(split, field) for field in ARRAY_FIELDS})
    path = Path(directory) / DATASET_FILE
    np.savez(path, **arrays)
    return path


def load_dataset(directory: str | Path) -> dict[str, Split]:
    """Inverse of save_dataset; accepts the run directory or the .npz itself."""
    path = Path(directory)
    if path.is_dir():
        path = path / DATASET_FILE
    with np.load(path, allow_pickle=False) as data:
        names = sorted({key.split(".", 1)[0] for key in data.files})
        return {
            name: Split(**{field: data[f"{name}.{field}"] for field in ARRAY_FIELDS})
            for name in names
        }
