import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Headless backend before any pyplot import, so plotting tests never open windows.
os.environ["MPLBACKEND"] = "Agg"

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dynamics.params import QuadParams  # noqa: E402


@pytest.fixture
def params() -> QuadParams:
    """Test vehicle, independent of the placeholder config so tests stay stable.

    Asymmetric inertia so torque-free tests exercise the gyroscopic term.
    """
    return QuadParams(
        mass=1.5,
        inertia=np.diag([0.01, 0.02, 0.03]),
        drag_coeffs=np.array([0.1, 0.2, 0.3]),
        gravity=9.81,
    )
