"""Test the `_potential_energy` module of `skcomplex.systems.physics.common.particles`."""

import numpy as np

from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.systems.physics.common.particles import MassPointElement, PotentialEnergyInteraction


def test_potential_energy_changes_the_velocity_by_the_force_over_the_mass():
    """Test that the energy `k x**2 / 2` makes a particle of mass `m` oscillate with the frequency `sqrt(k / m)`."""
    trap = PotentialEnergyInteraction(
        'trap',
        target='ball',
        potential=lambda s: 2.0 * s.target.x.dot_product(s.target.x),
    )
    system = CompositeSystem(
        elements=[MassPointElement('ball', x=[1.0], m=4.0)],
        interactions=[trap],
        space=ContinuousSpace(n_dim=1),
    )
    solver = ScipyODESolver(rtol=1e-10, atol=1e-12)
    trajectory = simulate(system, until=4.0, solver=solver, dt=0.1)
    np.testing.assert_allclose(trajectory.values('ball.x')[:, 0], np.cos(trajectory.times), atol=1e-7)
