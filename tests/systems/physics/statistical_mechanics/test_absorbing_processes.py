"""Test the `_absorbing_processes` module of `skcomplex.systems.physics.statistical_mechanics`."""

import numpy as np

from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.systems.physics.statistical_mechanics import AbsorbingBirthDeathSystem


def test_absorbing_birth_death_metastable_and_absorbed():
    """Test that the active state fluctuates around `rho_0` with the width of Bertin and that below `nu` it dies out."""
    active = (
        AbsorbingBirthDeathSystem(kappa=2.0, backend=NumbaBackend())
        .prepare()
        .simulate(until=1000, dt=0.5, random_state=0)
    )
    density = active.get_values('particles.n')[100:] / 100
    assert abs(density.mean() - 1.0) < 0.05
    assert abs(density.std() / density.mean() - np.sqrt(2.0 / 100)) < 0.02
    absorbed = AbsorbingBirthDeathSystem(kappa=0.5, backend=NumbaBackend()).prepare().simulate(dt=0.5, random_state=0)
    assert absorbed.get_values('particles.n')[-1] == 0
