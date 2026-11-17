"""Test the `_brownian` module of `skcomplex.systems.physics.soft_matter`."""

import numpy as np
import pytest

from skcomplex.systems.physics.soft_matter import BrownianNearWallSystem, StochasticResonanceSystem


@pytest.mark.slow
def test_stochastic_resonance_hops_between_wells():
    """Test that noise moves the particle between the wells of the double well."""
    values = StochasticResonanceSystem().prepare().simulate(until=300, random_state=0).get_values('particle.x')
    assert values.min() < -0.5
    assert values.max() > 0.5


def test_brownian_particle_stays_off_the_wall():
    """Test that the particle near the reflecting wall keeps a positive distance and finite values."""
    values = BrownianNearWallSystem().prepare().simulate(random_state=0).get_values('particle.z')
    assert np.all(np.isfinite(values))
    assert values.min() >= 0.0
