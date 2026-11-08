"""Test the `_instability` module of `skcomplex.systems.biology.ecology`."""

import numpy as np

from skcomplex.systems.biology.ecology import IntrinsicInstabilitySystem

from ...conftest import _series


def test_intrinsic_instability_leaves_its_fixed_point():
    """Test that the resident population stays near 1 until the invader grows and then declines."""
    trajectory = _series(IntrinsicInstabilitySystem(), 4000)
    x, y = trajectory.values('populations.x'), trajectory.values('populations.y')
    seeded = int(np.argmax(y > 0))
    assert 0 < seeded < 3000
    assert np.all(np.abs(x[100:seeded] - 1) < 0.15)
    assert x[-1] < 0.5
    assert y[-1] > 1
