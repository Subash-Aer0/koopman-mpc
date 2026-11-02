"""Quaternion utilities: multiplication (⊗), rotation (⊙), conversions."""


def quat_multiply(q1, q2):
    raise NotImplementedError


def quat_rotate(q, v):
    raise NotImplementedError


def quat_to_rotation_matrix(q):
    raise NotImplementedError


def quat_to_euler(q):
    """Return (roll, pitch, yaw); used for the 9-dim error state."""
    raise NotImplementedError
