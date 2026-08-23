import numpy as np
import pytest

from openvla_general_policy import map_physical_action_to_osc


def test_physical_action_mapping_is_bounded_and_maps_gripper():
    result = map_physical_action_to_osc([0.06, -0.07, 0.03, 0.5, -0.5, 0.25, 1.0])
    np.testing.assert_allclose(result, [1, -1, 0.5, 1, -1, 0.5, 1])
    assert np.all(result <= 1)
    assert np.all(result >= -1)


def test_action_dimension_is_checked():
    with pytest.raises(ValueError, match="7 values"):
        map_physical_action_to_osc([0, 1])
