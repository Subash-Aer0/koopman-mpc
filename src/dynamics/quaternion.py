"""Rotation utility functions.

Conventions (used throughout the repo):
    - Hamilton product, scalar-first order q = [qw, qx, qy, qz].
    - Unit quaternion q maps body-frame vectors to the world frame:
      v_world = q ⊙ v_body = q ⊗ [0, v_body] ⊗ q* = R(q) v_body.
    - Body frame FRD (x forward, y right, z down); world frame NED.
    - Euler angles are ZYX (yaw ψ, then pitch θ, then roll φ), in radians.
"""

import numpy as np

from ._validation import as_finite_vector

_UNIT_NORM_TOL = 1e-6
_ZERO_NORM_TOL = 1e-12


class Quaternion:
    """
    Implements quaternion based operations

    The product, conjugate and inverse work for any quaternion (pure
    quaternions such as [0, ω] included). Methods that interpret q as a
    rotation (to_rotation_matrix, to_euler, rotate) require a unit quaternion.
    """

    def __init__(self, q):

        q = as_finite_vector("q", q, 4)
        self.w = q[0]
        self.x = q[1]
        self.y = q[2]
        self.z = q[3]

    def __repr__(self):
        return f"Quaternion([{self.w}, {self.x}, {self.y}, {self.z}])"

    @staticmethod
    def identity() -> "Quaternion":
        """
        Returns the identity rotation [1, 0, 0, 0]
        """

        return Quaternion(np.array([1.0, 0.0, 0.0, 0.0]))

    def get_quaternion_vec(self):
        """
        Returns the vector part of the quaternion

        Returns
        -------
        qv: ndarray
            [qx, qy, qz]
        """

        return np.array([self.x, self.y, self.z])

    def to_numpy(self):
        """
        Returns the quaternion as ndarray

        Returns
        -------
        q: ndarray
            [qw, qx, qy, qz]
        """

        return np.array([self.w, self.x, self.y, self.z])

    def norm(self) -> float:
        """
        Returns the Euclidean norm ||q||
        """

        return float(np.linalg.norm(self.to_numpy()))

    def conjugate(self):
        """
        Returns the quaternion conjugate q = [w, x, y, z]

        Returns
        -------
        q* : Quaternion
             [w, -x, -y, -z]
        """

        q_conj = np.array([self.w, -self.x, -self.y, -self.z])

        return Quaternion(q_conj)

    def inverse(self):
        """
        Returns the quaternion inverse q⁻¹ = q* / ||q||², so q ⊗ q⁻¹ = [1, 0, 0, 0]

        For a unit quaternion this equals the conjugate.

        Raises
        ------
        ValueError
            If q is (numerically) zero.
        """

        norm = self.norm()
        if norm < _ZERO_NORM_TOL:
            raise ValueError("cannot invert a zero quaternion")
        return Quaternion(self.conjugate().to_numpy() / norm**2)

    def normalize(self):
        """
        Normalize quaternion to unit length

        Used to remove integration drift; never called implicitly by the
        product.

        Raises
        ------
        ValueError
            If q is (numerically) zero.
        """

        norm = self.norm()
        if norm < _ZERO_NORM_TOL:
            raise ValueError("cannot normalize a zero quaternion")
        return Quaternion(self.to_numpy() / norm)

    def _require_unit(self):
        if abs(self.norm() - 1.0) > _UNIT_NORM_TOL:
            raise ValueError(
                f"rotation requires a unit quaternion, got norm {self.norm()}"
            )

    def to_rotation_matrix(self):
        """
        Convert quaternion to rotation matrix R(q) ∈ SO(3), body -> world.

        Returns
        -------
        np.ndarray
            Rotation matrix (3, 3)
        """

        self._require_unit()
        qw, qx, qy, qz = self.w, self.x, self.y, self.z

        R = np.array(
            [
                [
                    qw**2 + qx**2 - qy**2 - qz**2,
                    2 * (qx * qy - qw * qz),
                    2 * (qx * qz + qw * qy),
                ],
                [
                    2 * (qx * qy + qw * qz),
                    qw**2 - qx**2 + qy**2 - qz**2,
                    2 * (qy * qz - qw * qx),
                ],
                [
                    2 * (qx * qz - qw * qy),
                    2 * (qy * qz + qw * qx),
                    qw**2 - qx**2 - qy**2 + qz**2,
                ],
            ]
        )

        return R

    @staticmethod
    def from_rotation_matrix(R: np.ndarray) -> "Quaternion":
        """
        Convert a rotation matrix (body -> world) to a unit quaternion [w, x, y, z]

        Raises
        ------
        ValueError
            If R is not a finite 3x3 matrix in SO(3).
        """

        R = np.asarray(R, dtype=float)
        if R.shape != (3, 3):
            raise ValueError(f"Rotation matrix must be 3x3, got {R.shape}")
        if not np.all(np.isfinite(R)):
            raise ValueError(f"Rotation matrix contains NaN or inf: {R}")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or np.linalg.det(R) <= 0:
            raise ValueError("Rotation matrix must be orthonormal with det +1")

        # Shepperd's method: branch on the largest of (trace, diagonal) so the
        # divisor s stays well away from zero.
        trace = float(np.trace(R))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            diag = np.diag(R)
            idx = int(np.argmax(diag))
            if idx == 0:
                s = np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0)) * 2.0
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif idx == 1:
                s = np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0)) * 2.0
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0)) * 2.0
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s

        return Quaternion(np.array([qw, qx, qy, qz])).normalize()

    def to_euler(self):
        """
        Convert to ZYX Euler angles

        Used for the 9-dim tracking-error state (Sec. III-A). Pitch is clipped
        to [-π/2, π/2]; roll and yaw are ill-defined at |θ| = π/2 (gimbal lock).

        Returns
        -------
        euler: ndarray
            [roll φ, pitch θ, yaw ψ], rad
        """

        self._require_unit()
        w, x, y, z = self.w, self.x, self.y, self.z
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([roll, pitch, yaw])

    @staticmethod
    def from_euler(euler) -> "Quaternion":
        """
        Inverse of to_euler

        Params
        ------
        euler: array_like
            [roll φ, pitch θ, yaw ψ], rad (ZYX)

        Returns
        -------
        q: Quaternion
            unit quaternion, body -> world
        """

        roll, pitch, yaw = as_finite_vector("euler", euler, 3)
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        return Quaternion(
            np.array(
                [
                    cr * cp * cy + sr * sp * sy,
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy,
                ]
            )
        )

    @staticmethod
    def multiply(q1: "Quaternion", q2: "Quaternion") -> "Quaternion":
        """
        Multiply two quaternions (Hamilton product)

        The result is not normalized: ||q1 ⊗ q2|| = ||q1|| ||q2||. This matters
        for pure quaternions, e.g. q̇ = ½ q ⊗ [0, ω] in Eq. (2).

        Params
        ------
        q1: Quaternion
        q2: Quaternion

        Returns
        -------
        q: Quaternion
            q1 ⊗ q2
        """

        q1_vec = q1.get_quaternion_vec()
        q2_vec = q2.get_quaternion_vec()

        q_vec = q1.w * q2_vec + q2.w * q1_vec + np.cross(q1_vec, q2_vec)

        q = np.array(
            [
                q1.w * q2.w - np.dot(q1_vec, q2_vec),
                *q_vec,
            ]
        )
        return Quaternion(q)

    @staticmethod
    def rotate(x, q: "Quaternion"):
        """
        Rotate vector x by unit quaternion q: q ⊙ x = q ⊗ [0, x] ⊗ q*

        Params
        ------
        x: array_like
            vector (3,), body frame
        q: Quaternion
            unit quaternion, body -> world

        Returns
        -------
        x_rotated: ndarray
            vector (3,), world frame, same magnitude as x
        """

        q._require_unit()
        x = as_finite_vector("x", x, 3)
        x_quat = Quaternion(np.array([0.0, *x]))
        x_rotated_quat = Quaternion.multiply(
            q, Quaternion.multiply(x_quat, q.conjugate())
        )

        x_rotated = x_rotated_quat.get_quaternion_vec()
        return x_rotated
