"""Test the `_xy_sampling` module of `skcomplex.systems.physics.common.spins`."""

import numpy as np

from skcomplex.systems.physics.common.spins import XYSamplingInteraction

from .conftest import _run


def test_aligned_xy_spins_stay_aligned_when_cold():
    """Test that aligned planar spins keep a large magnetization at a low temperature."""
    angles = _run(XYSamplingInteraction('xy', target='sites', width=0.5), 'theta', {'temperature': 0.1})
    assert np.abs(np.exp(1j * angles[-1]).mean()) > 0.9
