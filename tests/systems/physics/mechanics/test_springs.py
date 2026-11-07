"""Test the `_springs` module of `skcomplex.systems.physics.mechanics`."""

import numpy as np

from skcomplex.systems.physics.mechanics import TwoMassSpringDamperSystem

from ..conftest import _run, _series


def test_two_mass_spring_damper_conserves_energy_without_damping():
    """Test that the two masses between walls keep their energy without dampers and forces."""
    trajectory = _run(TwoMassSpringDamperSystem(k=2.0), 20.0, 0.1)
    x1, x2, v1, v2 = (array[:, 0] for array in _series(trajectory, 'first.x', 'second.x', 'first.x_d1', 'second.x_d1'))
    energy = 0.5 * (v1**2 + v2**2) + 0.5 * (x1 - 1) ** 2 + 0.5 * (x2 - 2) ** 2 + (x2 - x1 - 1) ** 2
    assert np.ptp(energy) < 1e-8
