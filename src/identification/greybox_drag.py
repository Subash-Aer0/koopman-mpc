"""Grey-box identification of the body drag coefficients from flight data.

The model structure is the physics of dynamics/drag.py with unknown D:
    v̇ = g e₃ − c R e₃ − (1/m) R diag(d) φ(v_rb),   φ = v_rb ‖v_rb‖,  v_rb = Rᵀ (v − v_w)
Everything except d is measured or known (v, η, c, v_w; m, g), and d enters
linearly, so d follows from (non-negative) linear least squares.

Discretization: commands and wind are held over each sample [t_k, t_k+1), so
v̇ is replaced by (v_k+1 − v_k)/dt, and R and φ by their averages at k and k+1
(midpoint rule). Thrust is the commanded c, i.e. the actuator is assumed ideal;
any thrust error is absorbed into d, mostly into d_z (body z).

Frames and units: world NED, body FRD, ZYX Euler angles in rad, v and v_w in
m/s, c in m/s² (mass-normalized), d in N·s²/m².
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from controllers.hover_model import rotation_zyx
from dynamics._validation import as_finite_vector

# Which coefficients are free: d = S θ, θ fitted. Rows are body axes x, y, z.
FORMS = {
    "anisotropic": np.eye(3),                                  # [d_x, d_y, d_z]
    "horizontal_equal": np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),   # [d_xy, d_z]
    "isotropic": np.ones((3, 1)),                              # [d]
}


def drag_regression(x9, u, v_w, dt: float, mass: float, gravity: float):
    """Regressor and target of one episode: y_k = X_k d, per sample k → k+1.

    Args:
        x9: [p, v, η] per sample, world NED / ZYX Euler (m, m/s, rad), (T, 9).
        u: Applied command [ω (rad/s), c (m/s²)] held over each sample, (T, 4).
        v_w: Wind held over each sample, world NED, m/s, (T, 3).
        dt: Sample time, s.
        mass: m, kg.
        gravity: g, m/s².

    Returns:
        X (T−1, 3, 3) with X_k[:, i] = −(1/m) R̄_k[:, i] φ̄_k,i, and
        y (T−1, 3) = (v_k+1 − v_k)/dt − g e₃ + c_k R̄_k e₃, both in m/s².
    """
    x9, u, v_w = (np.asarray(a, dtype=float) for a in (x9, u, v_w))
    if x9.ndim != 2 or x9.shape[1] != 9 or u.shape != (len(x9), 4) or v_w.shape != (len(x9), 3):
        raise ValueError(f"need x9 (T, 9), u (T, 4), v_w (T, 3); got {x9.shape}, {u.shape}, "
                         f"{v_w.shape}")
    if len(x9) < 2:
        raise ValueError("an episode needs at least 2 samples")
    if not (np.all(np.isfinite(x9)) and np.all(np.isfinite(u)) and np.all(np.isfinite(v_w))):
        raise ValueError("x9, u or v_w contains NaN or inf")
    for name, value in (("dt", dt), ("mass", mass), ("gravity", gravity)):
        if not (np.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be positive and finite, got {value}")

    R = np.array([rotation_zyx(eta) for eta in x9[:, 6:9]])            # (T, 3, 3)
    v = x9[:, 3:6]
    # φ at both ends of each sample, with the wind held over the sample.
    phi = np.empty((len(x9) - 1, 2, 3))
    for end in (0, 1):
        v_rb = np.einsum("kji,kj->ki", R[end:len(R) - 1 + end], v[end:len(v) - 1 + end] - v_w[:-1])
        phi[:, end] = v_rb * np.linalg.norm(v_rb, axis=1, keepdims=True)
    R_mid = 0.5 * (R[:-1] + R[1:])
    phi_mid = phi.mean(axis=1)
    X = -R_mid * phi_mid[:, None, :] / mass
    y = (v[1:] - v[:-1]) / dt - np.array([0.0, 0.0, gravity]) + u[:-1, 3:4] * R_mid[:, :, 2]
    return X, y


def fit_drag(X, y, form: str = "anisotropic") -> np.ndarray:
    """Non-negative least squares for d (3,), N·s²/m², given stacked X (K, 3, 3), y (K, 3).

    Args:
        X, y: drag_regression outputs, concatenated over episodes.
        form: A key of FORMS; which coefficients are tied together.

    Raises:
        ValueError: For an unknown form, mismatched shapes or too few samples.
    """
    if form not in FORMS:
        raise ValueError(f"form must be one of {sorted(FORMS)}, got {form!r}")
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 3 or X.shape[1:] != (3, 3) or y.shape != X.shape[:2]:
        raise ValueError(f"need X (K, 3, 3) and y (K, 3), got {X.shape} and {y.shape}")
    S = FORMS[form]
    A = (X @ S).reshape(-1, S.shape[1])
    if A.shape[0] < 10 * S.shape[1]:
        raise ValueError(f"{A.shape[0]} equations are too few for {S.shape[1]} coefficients")
    theta = lsq_linear(A, y.reshape(-1), bounds=(0.0, np.inf)).x   # drag cannot push forward
    return S @ theta


def residual_rms(X, y, drag_coeffs) -> np.ndarray:
    """RMS of y − X d per world axis, m/s², shape (3,)."""
    d = as_finite_vector("drag_coeffs", drag_coeffs, 3)
    return np.sqrt(np.mean((np.asarray(y) - np.asarray(X) @ d) ** 2, axis=0))


# --- run record -----------------------------------------------------------------

RESULT_FILE = "greybox_drag.json"


def save_greybox_drag(run_dir, result: dict) -> Path:
    """Write a fit result (with key "drag_coeffs") to <run_dir>/greybox_drag.json."""
    as_finite_vector("drag_coeffs", result["drag_coeffs"], 3)
    path = Path(run_dir) / RESULT_FILE
    path.write_text(json.dumps(result, indent=2))
    return path


def load_greybox_drag(run_dir) -> np.ndarray:
    """Fitted drag coefficients (3,), N·s²/m², from a fit_greybox_drag run directory."""
    result = json.loads((Path(run_dir) / RESULT_FILE).read_text())
    d = as_finite_vector("drag_coeffs", result["drag_coeffs"], 3)
    if np.any(d < 0):
        raise ValueError(f"drag coefficients must be non-negative, got {d}")
    return d
