"""Test the `_brownian` module of `skcomplex.models.physics.soft_matter`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models.physics.soft_matter import brownian_near_wall, stochastic_resonance


@pytest.mark.slow
def test_stochastic_resonance_hops_between_wells():
    """Test that noise moves the particle between the wells of the double well."""
    values = (
        Simulation(stochastic_resonance(), dt=0.5, random_state=0)
        .check()
        .prepare()
        .run(until=300)
        .trajectory_.values('particle.x')
    )
    assert values.min() < -0.5
    assert values.max() > 0.5


def test_brownian_particle_stays_off_the_wall():
    """Test that the particle near the reflecting wall keeps a positive distance and finite values."""
    values = (
        Simulation(brownian_near_wall(), dt=0.1, random_state=0)
        .check()
        .prepare()
        .run(until=10)
        .trajectory_.values('particle.z')
    )
    assert np.all(np.isfinite(values))
    assert values.min() >= 0.0
