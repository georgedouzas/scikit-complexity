"""Test the `_spin_sampling` module of `skcomplex.models.physics.common.spins`."""

import numpy as np

from skcomplex.core.laws.utils import neighbors
from skcomplex.models.physics.common.spins import SpinSamplingInteraction

from .conftest import _run


def _antiferromagnet(namespace):
    """Return the local energy of an antiferromagnetic spin."""
    return namespace.target.spin * neighbors(namespace.target.other.spin)


def test_sampling_runs_at_the_temperature_of_the_spins():
    """Test that cold spins order into the checkerboard of an antiferromagnet and hot spins do not."""
    sampling = SpinSamplingInteraction('sampling', target='sites', state='spin', energy=_antiferromagnet)
    checkerboard = np.where(np.indices((12, 12)).sum(axis=0) % 2 == 0, 1.0, -1.0).ravel()
    cold = _run(sampling, 'spin', {'spin': checkerboard, 'temperature': 0.5})
    hot = _run(sampling, 'spin', {'spin': checkerboard, 'temperature': 20.0})
    assert (cold[-1] * checkerboard).mean() > 0.9
    assert abs((hot[-20:] * checkerboard).mean()) < 0.3
