"""Reference trajectories x_r(t) (two 14 s trajectories for training).

References are callables t -> (x_ref, u_ref) as expected by
simulation.closed_loop.simulate: x_ref = [p, q, v], shape (10,), and
u_ref = [ω, c], shape (4,), in the frames and units of
controllers/sdlqr_model.py (NED world, FRD body, rad/s, m/s²).
"""
import numpy as np

from dynamics._validation import as_finite_vector
from dynamics.quaternion import Quaternion


class SetpointReference:
    """Constant position and yaw, level attitude, at rest; u_ref is hover thrust."""

    def __init__(self, position, yaw: float, gravity: float):
        """
        Args:
            position: Setpoint p, NED world frame, m, shape (3,).
            yaw: Heading ψ, rad (ZYX Euler, 0 = north).
            gravity: g, m/s²; hover needs c = g.
        """
        position = as_finite_vector("position", position, 3)
        if not np.isfinite(yaw):
            raise ValueError(f"yaw must be finite, got {yaw}")
        if not (np.isfinite(gravity) and gravity > 0):
            raise ValueError(f"gravity must be positive and finite, got {gravity}")

        q = Quaternion.from_euler([0.0, 0.0, yaw]).to_numpy()
        self._x_ref = np.concatenate([position, q, np.zeros(3)])
        self._u_ref = np.array([0.0, 0.0, 0.0, gravity])

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        return self._x_ref.copy(), self._u_ref.copy()


def flat_state(p, v, a, j, yaw: float, gravity: float) -> tuple[np.ndarray, np.ndarray]:
    """Differential flatness: (p, v, a, j, ψ) -> (x_ref, u_ref) for the LQR model.

    From v̇ = g e₃ − c R e₃ (NED, thrust along −z_body, sdlqr_model.py):
        F  = g e₃ − a = c R e₃,   c = ‖F‖,   b₃ = F / c          (body z, down at hover)
        b₂ = (b₃ × x_ψ) / ‖b₃ × x_ψ‖,  x_ψ = [cos ψ, sin ψ, 0],   b₁ = b₂ × b₃
        R  = [b₁ b₂ b₃]  (body FRD -> world NED)
    Body rates from Ṙ = R [ω]×, with constant yaw (ψ̇ = 0):
        ḃ₃ = −(I − b₃b₃ᵀ) j / c          ω_x = −b₂·ḃ₃,   ω_y = b₁·ḃ₃
        ḃ₂ = (I − b₂b₂ᵀ)(ḃ₃ × x_ψ) / ‖b₃ × x_ψ‖     ω_z = −b₁·ḃ₂

    Args:
        p, v, a, j: Position (m), velocity (m/s), acceleration (m/s²) and
            jerk (m/s³), NED world frame, shape (3,) each.
        yaw: Constant heading ψ, rad.
        gravity: g, m/s².

    Returns:
        x_ref = [p, q, v], shape (10,), and u_ref = [ω (rad/s), c (m/s²)], shape (4,).

    Raises:
        ValueError: If the commanded acceleration needs zero thrust (free fall)
            or puts the body z axis along the heading (yaw undefined).
    """
    F = np.array([0.0, 0.0, gravity]) - a
    c = np.linalg.norm(F)
    if c < 1e-6:
        raise ValueError("trajectory demands free fall; thrust direction is undefined")
    b3 = F / c

    x_yaw = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    n = np.cross(b3, x_yaw)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-6:
        raise ValueError("body z axis is parallel to the heading; yaw is undefined")
    b2 = n / n_norm
    b1 = np.cross(b2, b3)
    R = np.column_stack([b1, b2, b3])

    b3_dot = -(j - b3 * (b3 @ j)) / c
    n_dot = np.cross(b3_dot, x_yaw)
    b2_dot = (n_dot - b2 * (b2 @ n_dot)) / n_norm
    omega = np.array([-b2 @ b3_dot, b1 @ b3_dot, -b1 @ b2_dot])

    q = Quaternion.from_rotation_matrix(R).to_numpy()
    if q[0] < 0:  # one sign convention along the trajectory (q and −q are the same rotation)
        q = -q
    return np.concatenate([p, q, v]), np.array([*omega, c])


class Figure8Reference:
    """Non-planar figure-8 with flat feedforward, constant yaw.

    NED world frame, Ω = 2π / period:
        x = c_x + A sin(Ωt)
        y = c_y + B sin(2Ωt)     (the 8 in the horizontal plane)
        z = c_z − C cos(Ωt)      (cos, not sin, so z is not ∝ x and the curve is not planar)
    """

    def __init__(self, center, amplitudes, period: float, yaw: float, gravity: float):
        """
        Args:
            center: [c_x, c_y, c_z], NED world frame, m.
            amplitudes: [A, B, C], m.
            period: Time for one full 8, s.
            yaw: Constant heading ψ, rad.
            gravity: g, m/s².
        """
        self.center = as_finite_vector("center", center, 3)
        self.amplitudes = as_finite_vector("amplitudes", amplitudes, 3)
        if not (np.isfinite(period) and period > 0):
            raise ValueError(f"period must be positive and finite, got {period}")
        if not np.isfinite(yaw):
            raise ValueError(f"yaw must be finite, got {yaw}")
        if not (np.isfinite(gravity) and gravity > 0):
            raise ValueError(f"gravity must be positive and finite, got {gravity}")
        self.omega = 2.0 * np.pi / period
        self.yaw = float(yaw)
        self.gravity = float(gravity)

    def derivatives(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Position, velocity, acceleration and jerk at t, NED world frame."""
        A, B, C = self.amplitudes
        w = self.omega
        s1, c1 = np.sin(w * t), np.cos(w * t)
        s2, c2 = np.sin(2 * w * t), np.cos(2 * w * t)
        p = self.center + np.array([A * s1, B * s2, -C * c1])
        v = np.array([A * w * c1, 2 * B * w * c2, C * w * s1])
        a = np.array([-A * w**2 * s1, -4 * B * w**2 * s2, C * w**2 * c1])
        j = np.array([-A * w**3 * c1, -8 * B * w**3 * c2, -C * w**3 * s1])
        return p, v, a, j

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        return flat_state(*self.derivatives(t), self.yaw, self.gravity)


class CircleReference:
    """Horizontal circle with a vertical oscillation, flat feedforward, constant yaw.

    NED world frame, Ω = 2π / period:
        x = c_x + r sin(Ωt)
        y = c_y − r cos(Ωt)
        z = c_z − h sin(2Ωt)     (twice per lap, so the curve is not planar for h > 0)
    """

    def __init__(self, center, radius: float, vertical_amplitude: float, period: float,
                 yaw: float, gravity: float):
        """
        Args:
            center: [c_x, c_y, c_z], NED world frame, m.
            radius: r, m, > 0.
            vertical_amplitude: h, m, ≥ 0; 0 gives a level circle.
            period: Time for one lap, s.
            yaw: Constant heading ψ, rad.
            gravity: g, m/s².
        """
        self.center = as_finite_vector("center", center, 3)
        if not (np.isfinite(radius) and radius > 0):
            raise ValueError(f"radius must be positive and finite, got {radius}")
        if not (np.isfinite(vertical_amplitude) and vertical_amplitude >= 0):
            raise ValueError(f"vertical_amplitude must be non-negative and finite, "
                             f"got {vertical_amplitude}")
        if not (np.isfinite(period) and period > 0):
            raise ValueError(f"period must be positive and finite, got {period}")
        if not np.isfinite(yaw):
            raise ValueError(f"yaw must be finite, got {yaw}")
        if not (np.isfinite(gravity) and gravity > 0):
            raise ValueError(f"gravity must be positive and finite, got {gravity}")
        self.radius = float(radius)
        self.vertical_amplitude = float(vertical_amplitude)
        self.omega = 2.0 * np.pi / period
        self.yaw = float(yaw)
        self.gravity = float(gravity)

    def derivatives(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Position, velocity, acceleration and jerk at t, NED world frame."""
        r, h, w = self.radius, self.vertical_amplitude, self.omega
        s1, c1 = np.sin(w * t), np.cos(w * t)
        s2, c2 = np.sin(2 * w * t), np.cos(2 * w * t)
        p = self.center + np.array([r * s1, -r * c1, -h * s2])
        v = np.array([r * w * c1, r * w * s1, -2 * h * w * c2])
        a = np.array([-r * w**2 * s1, r * w**2 * c1, 4 * h * w**2 * s2])
        j = np.array([-r * w**3 * c1, -r * w**3 * s1, 8 * h * w**3 * c2])
        return p, v, a, j

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        return flat_state(*self.derivatives(t), self.yaw, self.gravity)


class SumOfSinesReference:
    """Random smooth reference: per-axis sums of sinusoids, flat feedforward, constant yaw.

    NED world frame, per axis i and component k (ω_ki = 2π f_ki):
        p_i(t) = c_i + Σ_k a_ki sin(ω_ki t + φ_ki)
    Used for trajectory-agnostic identification data: unlike a circle, speed,
    acceleration and their relative direction vary independently.
    """

    def __init__(self, center, amplitudes, frequencies, phases, yaw: float, gravity: float):
        """
        Args:
            center: [c_x, c_y, c_z], NED world frame, m.
            amplitudes: a, m, shape (K, 3), one column per axis.
            frequencies: f, Hz, shape (K, 3), > 0.
            phases: φ, rad, shape (K, 3).
            yaw: Constant heading ψ, rad.
            gravity: g, m/s².
        """
        self.center = as_finite_vector("center", center, 3)
        arrays = {}
        for name, value in (("amplitudes", amplitudes), ("frequencies", frequencies),
                            ("phases", phases)):
            value = np.asarray(value, dtype=float)
            if value.ndim != 2 or value.shape[1] != 3 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite (K, 3) array, got shape {value.shape}")
            arrays[name] = value
        if not arrays["amplitudes"].shape == arrays["frequencies"].shape == arrays["phases"].shape:
            raise ValueError("amplitudes, frequencies and phases must have the same shape")
        if np.any(arrays["frequencies"] <= 0):
            raise ValueError("frequencies must be positive")
        if not np.isfinite(yaw):
            raise ValueError(f"yaw must be finite, got {yaw}")
        if not (np.isfinite(gravity) and gravity > 0):
            raise ValueError(f"gravity must be positive and finite, got {gravity}")
        self.amplitudes = arrays["amplitudes"]
        self.omegas = 2.0 * np.pi * arrays["frequencies"]
        self.phases = arrays["phases"]
        self.yaw = float(yaw)
        self.gravity = float(gravity)

    def derivative_bounds(self) -> np.ndarray:
        """Per-axis bounds on |dⁿp/dtⁿ| over all t, n = 0..3: Σ_k |a_k| ω_kⁿ, shape (4, 3)."""
        return np.array([np.sum(np.abs(self.amplitudes) * self.omegas**n, axis=0)
                         for n in range(4)])

    def derivatives(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Position, velocity, acceleration and jerk at t, NED world frame."""
        a, w = self.amplitudes, self.omegas
        s, c = np.sin(w * t + self.phases), np.cos(w * t + self.phases)
        p = self.center + np.sum(a * s, axis=0)
        v = np.sum(a * w * c, axis=0)
        acc = -np.sum(a * w**2 * s, axis=0)
        j = -np.sum(a * w**3 * c, axis=0)
        return p, v, acc, j

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        return flat_state(*self.derivatives(t), self.yaw, self.gravity)


def sample_sum_of_sines(rng: np.random.Generator, center, n_components: int,
                        frequency_range, peak_tilt: float, max_speed: float,
                        vertical_amplitude: float, yaw: float,
                        gravity: float) -> SumOfSinesReference:
    """Draw a SumOfSinesReference whose tilt and speed are bounded for all t.

    Frequencies f ~ U(frequency_range), phases φ ~ U(0, 2π), raw amplitudes
    ~ U(0, 1); the amplitudes are then scaled with the guaranteed bounds
    b_n = Σ_k |a_k| ω_kⁿ (derivative_bounds), so that for all t:
        vertical:    |p_z − c_z| ≤ vertical_amplitude
        tilt:        atan(‖b_2,xy‖ / (g − b_2,z)) ≤ peak_tilt   (thrust direction of the
                     flat feedforward), reached when the bound is tight
        speed:       ‖b_1,xy‖ ≤ max_speed  (horizontal amplitudes shrink if needed)
    These are bounds, so the reference's actual peak tilt is at most peak_tilt.

    Args:
        rng: Source of every draw.
        center: Mean position, NED world frame, m, shape (3,).
        n_components: K ≥ 1 sinusoids per axis.
        frequency_range: [f_min, f_max], Hz, 0 < f_min ≤ f_max.
        peak_tilt: Tilt bound, rad, in (0, π/2).
        max_speed: Horizontal speed bound, m/s, > 0.
        vertical_amplitude: Vertical excursion bound, m, ≥ 0.
        yaw: Constant heading ψ, rad.
        gravity: g, m/s².

    Raises:
        ValueError: For out-of-range arguments, or a vertical motion so fast
            that its acceleration bound reaches g (no thrust left for tilt).
    """
    f_min, f_max = (float(f) for f in frequency_range)
    if not (int(n_components) >= 1):
        raise ValueError(f"n_components must be ≥ 1, got {n_components}")
    if not (0 < f_min <= f_max and np.isfinite(f_max)):
        raise ValueError(f"frequency_range must satisfy 0 < f_min ≤ f_max, got {frequency_range}")
    if not (0 < peak_tilt < np.pi / 2):
        raise ValueError(f"peak_tilt must be in (0, π/2) rad, got {peak_tilt}")
    if not (np.isfinite(max_speed) and max_speed > 0):
        raise ValueError(f"max_speed must be positive and finite, got {max_speed}")
    if not (np.isfinite(vertical_amplitude) and vertical_amplitude >= 0):
        raise ValueError(f"vertical_amplitude must be non-negative, got {vertical_amplitude}")
    shape = (int(n_components), 3)
    frequencies = rng.uniform(f_min, f_max, shape)
    phases = rng.uniform(0.0, 2.0 * np.pi, shape)
    amplitudes = rng.uniform(0.0, 1.0, shape)
    omegas = 2.0 * np.pi * frequencies

    # Vertical: peak displacement Σ|a_z| = vertical_amplitude.
    amplitudes[:, 2] *= vertical_amplitude / np.sum(amplitudes[:, 2])
    accel_z = np.sum(amplitudes[:, 2] * omegas[:, 2]**2)
    if accel_z >= gravity:
        raise ValueError(f"vertical acceleration bound {accel_z:.2f} m/s² reaches g; "
                         "lower vertical_amplitude or the frequencies")
    # Horizontal: tilt bound first, then the speed bound if it is tighter.
    xy = slice(0, 2)
    accel_xy = np.linalg.norm(np.sum(amplitudes[:, xy] * omegas[:, xy]**2, axis=0))
    scale = (gravity - accel_z) * np.tan(peak_tilt) / accel_xy
    speed_xy = scale * np.linalg.norm(np.sum(amplitudes[:, xy] * omegas[:, xy], axis=0))
    scale *= min(1.0, max_speed / speed_xy)
    amplitudes[:, xy] *= scale
    return SumOfSinesReference(center, amplitudes, frequencies, phases, yaw, gravity)


def reference_trajectory(name, t):
    raise NotImplementedError
