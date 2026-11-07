"""Test the `_absorbing` module of `skcomplex.systems.physics.criticality`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.criticality import DirectedPercolationLangevinSystem


def _stationary_density(energy):
    """Return the late mean activity of the directed percolation field at a fixed energy."""
    system = DirectedPercolationLangevinSystem((16, 16), energy=energy)
    trajectory = simulate(system, until=40, random_state=0)
    return float(np.asarray(trajectory.values('neurons.rho'))[-10:].mean())


def test_directed_percolation_langevin_has_an_absorbing_threshold():
    """Test that the stationary activity vanishes below the threshold energy 1 and grows above it as (nu - 1)."""
    below = [_stationary_density(energy) for energy in (0.5, 0.8)]
    above = [_stationary_density(energy) for energy in (1.4, 1.8)]
    assert max(below) < 0.01
    assert above[0] < above[1]
    np.testing.assert_allclose(above, [0.4, 0.8], atol=0.06)
