"""Test the `_instability` module of `skcomplex.models.biology.ecology`."""

import numpy as np

from ...conftest import _series


def test_intrinsic_instability_leaves_its_fixed_point():
    """Test that the resident population stays near 1 until the invader grows and then declines."""
    trajectory = _series('intrinsic_instability', 4000)
    x, y = trajectory['populations.x'], trajectory['populations.y']
    seeded = int(np.argmax(y > 0))
    assert 0 < seeded < 3000
    assert np.all(np.abs(x[100:seeded] - 1) < 0.15)
    assert x[-1] < 0.5
    assert y[-1] > 1
