"""Test the `_absorbing` module of `skcomplex.models.physics.criticality`."""

import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyamaSolver
from skcomplex.models.physics.criticality import directed_percolation_langevin


def _stationary_density(energy):
    """Return the late mean activity of the directed percolation field at a fixed energy."""
    system = directed_percolation_langevin((16, 16), energy=energy)
    trajectory = (

            Simulation(system, dt=1.0, solver=EulerMaruyamaSolver(dt=0.01), random_state=0)
            .check()
            .run(until=40)
            .trajectory_

    )
    return float(np.asarray(trajectory.values('neurons.rho'))[-10:].mean())


def test_directed_percolation_langevin_has_an_absorbing_threshold():
    """Test that the stationary activity vanishes below the threshold energy 1 and grows above it as (nu - 1)."""
    below = [_stationary_density(energy) for energy in (0.5, 0.8)]
    above = [_stationary_density(energy) for energy in (1.4, 1.8)]
    assert max(below) < 0.01
    assert above[0] < above[1]
    np.testing.assert_allclose(above, [0.4, 0.8], atol=0.06)
