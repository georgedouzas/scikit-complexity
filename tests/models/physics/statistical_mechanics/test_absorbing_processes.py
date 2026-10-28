"""Test the `_absorbing_processes` module of `skcomplex.models.physics.statistical_mechanics`."""

import numpy as np

from skcomplex.engine.backends import NumbaBackend
from skcomplex.engine.runtime import Simulation
from skcomplex.models.physics.statistical_mechanics import (
    absorbing_birth_death,
)


def test_absorbing_birth_death_metastable_and_absorbed():
    """Test that the active state fluctuates around `rho_0` with the width of Bertin and that below `nu` it dies out."""
    active = Simulation(absorbing_birth_death(kappa=2.0), dt=0.5, random_state=0, backend=NumbaBackend()).check().run(
        until=1000,
    )
    density = active.trajectory_.values('particles.n')[100:] / 100
    assert abs(density.mean() - 1.0) < 0.05
    assert abs(density.std() / density.mean() - np.sqrt(2.0 / 100)) < 0.02
    absorbed = Simulation(absorbing_birth_death(kappa=0.5), dt=0.5, random_state=0, backend=NumbaBackend()).check().run(
        until=100,
    )
    assert absorbed.trajectory_.values('particles.n')[-1] == 0
