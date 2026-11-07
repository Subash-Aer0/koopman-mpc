"""Quaternion math against analytic cases."""
import numpy as np
import pytest

from dynamics.quaternion import Quaternion

IDENTITY = Quaternion.identity()


def assert_quat_close(q1, q2, atol=1e-12):
    np.testing.assert_allclose(q1.to_numpy(), q2.to_numpy(), atol=atol)


def test_identity_rotation_is_identity_matrix():
    np.testing.assert_allclose(IDENTITY.to_rotation_matrix(), np.eye(3))


def test_multiply_by_identity_is_noop():
    q = Quaternion.from_euler([0.1, -0.2, 0.3])
    assert_quat_close(Quaternion.multiply(q, IDENTITY), q)
    assert_quat_close(Quaternion.multiply(IDENTITY, q), q)


def test_quaternion_times_conjugate_is_identity():
    q = Quaternion.from_euler([0.4, 0.5, -1.2])
    assert_quat_close(Quaternion.multiply(q, q.conjugate()), IDENTITY)


def test_inverse_of_non_unit_quaternion():
    q = Quaternion([1.0, 2.0, -0.5, 0.3])
    assert_quat_close(Quaternion.multiply(q, q.inverse()), IDENTITY)


def test_positive_yaw_turns_nose_from_north_to_east():
    # NED: +90 deg yaw maps body x (forward) onto world y (east).
    q = Quaternion.from_euler([0.0, 0.0, np.pi / 2])
    np.testing.assert_allclose(Quaternion.rotate([1, 0, 0], q), [0, 1, 0], atol=1e-12)


def test_positive_pitch_raises_nose():
    # FRD/NED: +pitch tilts body x upward, i.e. towards world −z.
    q = Quaternion.from_euler([0.0, np.pi / 2, 0.0])
    np.testing.assert_allclose(Quaternion.rotate([1, 0, 0], q), [0, 0, -1], atol=1e-12)


def test_rotate_matches_rotation_matrix():
    q = Quaternion.from_euler([0.3, -0.2, 1.0])
    v = np.array([3.0, -1.0, 4.0])
    np.testing.assert_allclose(Quaternion.rotate(v, q), q.to_rotation_matrix() @ v, atol=1e-12)


def test_rotate_preserves_magnitude():
    # Regression: the pasted version normalized inside multiply, so rotate
    # returned a unit vector (|[3, 0, 4]| came back as 1 instead of 5).
    q = Quaternion.from_euler([0.3, -0.2, 1.0])
    assert np.isclose(np.linalg.norm(Quaternion.rotate([3.0, 0.0, 4.0], q)), 5.0)


def test_product_with_pure_quaternion_is_not_normalized():
    # Regression: q̇ = ½ q ⊗ [0, ω] needs ||q ⊗ [0, ω]|| = ||ω||, not 1.
    q = Quaternion.from_euler([0.3, -0.2, 1.0])
    omega = Quaternion([0.0, 0.0, 0.0, 0.5])
    assert np.isclose(Quaternion.multiply(q, omega).norm(), 0.5)


def test_rotation_matrix_is_in_so3():
    R = Quaternion.from_euler([0.7, -0.3, 2.0]).to_rotation_matrix()
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_product_composes_rotations():
    q1 = Quaternion.from_euler([0.2, 0.1, -0.4])
    q2 = Quaternion.from_euler([-0.5, 0.3, 0.9])
    np.testing.assert_allclose(
        Quaternion.multiply(q1, q2).to_rotation_matrix(),
        q1.to_rotation_matrix() @ q2.to_rotation_matrix(),
        atol=1e-12,
    )


@pytest.mark.parametrize("euler", [
    [0.0, 0.0, 0.0],
    [0.3, -0.2, 1.0],
    [-2.5, 1.2, -3.0],
])
def test_euler_round_trip(euler):
    np.testing.assert_allclose(Quaternion.from_euler(euler).to_euler(), euler, atol=1e-12)


@pytest.mark.parametrize("euler", [
    [0.3, -0.2, 1.0],        # trace > 0 branch
    [np.pi, 0.0, 0.0],       # 180 deg about x: largest diagonal R[0, 0]
    [0.0, 0.0, np.pi],       # 180 deg about z: largest diagonal R[2, 2]
    [np.pi, 0.0, np.pi],     # 180 deg about y: largest diagonal R[1, 1]
])
def test_rotation_matrix_round_trip(euler):
    R = Quaternion.from_euler(euler).to_rotation_matrix()
    np.testing.assert_allclose(
        Quaternion.from_rotation_matrix(R).to_rotation_matrix(), R, atol=1e-12
    )


def test_non_unit_quaternion_rejected_for_rotation():
    with pytest.raises(ValueError, match="unit quaternion"):
        Quaternion([1.0, 1.0, 0.0, 0.0]).to_rotation_matrix()


def test_zero_quaternion_normalize_raises():
    # Regression: the pasted version silently returned identity.
    with pytest.raises(ValueError, match="zero quaternion"):
        Quaternion([0.0, 0.0, 0.0, 0.0]).normalize()


def test_bad_inputs_rejected():
    with pytest.raises(ValueError, match="shape"):
        Quaternion([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="NaN"):
        Quaternion.rotate([np.nan, 0, 0], IDENTITY)
    with pytest.raises(ValueError, match="orthonormal"):
        Quaternion.from_rotation_matrix(2 * np.eye(3))
