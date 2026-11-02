"""Test the `_wells` module of `skcomplex.models.physics.mechanics`."""

import numpy as np
import pytest

from skcomplex.models.physics.mechanics import (
    particle_in_a_well,
)

from ..conftest import _run, _series


@pytest.mark.parametrize(('v', 'bounded'), [(0.5, True), (1.5, False)])
def test_particle_in_a_well_is_bound_below_the_barrier(v, bounded):
    """Test that the particle stays in the well below the barrier energy `4 k e^-2` and escapes above it."""
    trajectory = _run(particle_in_a_well(1.0, v), 20.0, 0.05)
    x, velocity = (array[:, 0] for array in _series(trajectory, 'particle.x', 'particle.x_d1'))
    energy = 0.5 * velocity**2 + 2 * x**2 * np.exp(-x)
    assert np.ptp(energy) < 1e-8
    assert (x.max() < 2) == bounded
