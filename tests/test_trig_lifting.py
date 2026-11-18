"""Trig lifting: reduces to the selective monomial lifting, recovers angles, avoids sin²+cos²=1."""
import numpy as np
import pytest

from koopman.selective_lifting import SelectiveMonomialLifting
from koopman.trig_lifting import TrigMonomialLifting

NAMES = ("p_x", "p_y", "p_z", "v_x", "v_y", "v_z", "roll", "pitch", "yaw")


def test_without_trig_it_equals_the_selective_lifting():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(20, 9))
    old = SelectiveMonomialLifting(9, 2, [3, 4, 5, 6, 7, 8])
    new = TrigMonomialLifting(9, 2, [3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(new(s), old(s))
    assert new.feature_names(NAMES) == old.feature_names(NAMES)
    assert new.constant_index == old.constant_index


def test_trig_features_use_the_angle_in_radians():
    scale = np.array([0.2, 0.1, 0.05])
    lifting = TrigMonomialLifting(9, 1, [3, 4, 5], trig_indices=[6, 7, 8], trig_scale=scale)
    angles = np.array([0.3, -0.2, 0.1])                       # rad
    s = np.zeros((1, 9))
    s[0, 6:9] = angles / scale
    psi = lifting(s)[0]
    names = lifting.feature_names(NAMES)
    assert psi[names.index("sin(roll)")] == pytest.approx(np.sin(0.3))
    assert psi[names.index("(cos(pitch)-1)")] == pytest.approx(np.cos(-0.2) - 1.0)
    np.testing.assert_array_equal(psi[:9], s[0])             # state first: C = [I 0]
    assert psi[lifting.constant_index] == 1.0


def test_level_attitude_gives_zero_trig_features():
    lifting = TrigMonomialLifting(9, 2, [3, 4, 5], [6, 7, 8], [0.1, 0.1, 0.1])
    psi = lifting(np.zeros((1, 9)))[0]
    assert np.count_nonzero(psi) == 1                         # only the constant


def test_no_square_of_cos_so_the_regressor_has_full_rank():
    """sin² + (cos−1)² = −2(cos−1): keeping (cos−1)² would make the features dependent."""
    lifting = TrigMonomialLifting(9, 2, [3, 4, 5], [6, 7, 8], [0.2, 0.2, 0.2])
    names = lifting.feature_names(NAMES)
    assert "(cos(roll)-1)*(cos(roll)-1)" not in names
    assert "sin(roll)*sin(roll)" in names and "sin(roll)*(cos(roll)-1)" in names
    psi = lifting(np.random.default_rng(1).uniform(-2.0, 2.0, size=(2000, 9)))
    assert np.linalg.matrix_rank(psi) == lifting.lifted_dim
    # 9 states + 1 + 6 trig + degree 2 over 9 base vars (45) − 3 cos² = 58
    assert lifting.lifted_dim == 58


@pytest.mark.parametrize("kwargs, message", [
    ({"nonlinear_indices": [3, 6], "trig_indices": [6], "trig_scale": [0.1]}, "raw or trig"),
    ({"nonlinear_indices": [3], "trig_indices": [6], "trig_scale": [0.0]}, "trig_scale"),
    ({"nonlinear_indices": [3], "trig_indices": [6]}, "trig_scale"),
    ({"nonlinear_indices": [3, 3]}, "distinct"),
])
def test_rejects_bad_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TrigMonomialLifting(9, 2, **kwargs)
