"""Test the `_ising_sampling` module of `skcomplex.systems.physics.common.spins`."""

import numpy as np

from skcomplex.systems.physics.common.spins import IsingSamplingInteraction

from .conftest import _run


def test_ising_orders_cold_and_disorders_hot():
    """Test that Ising spins stay magnetized far below the critical temperature and lose order far above it."""
    up = np.ones((12, 12))
    cold = _run(IsingSamplingInteraction('ising', target='sites'), 'spin', {'spin': up, 'temperature': 1.0})
    hot = _run(IsingSamplingInteraction('ising', target='sites'), 'spin', {'spin': up, 'temperature': 10.0})
    assert cold[-1].mean() > 0.9
    assert abs(hot[-20:].mean()) < 0.3


def test_field_aligns_the_spins():
    """Test that a strong field turns spins that start down at a high temperature."""
    down = -np.ones((12, 12))
    values = {'spin': down, 'h': 5.0, 'temperature': 2.0}
    spins = _run(IsingSamplingInteraction('ising', target='sites'), 'spin', values)
    assert spins[-1].mean() > 0.9
