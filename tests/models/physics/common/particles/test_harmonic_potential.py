"""Test the `_harmonic_potential` module of `skcomplex.models.physics.common.particles`."""

import numpy as np

from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models.physics.common.particles import HarmonicPotentialInteraction, MassPointElement


def test_harmonic_potential_oscillates_about_its_center():
    """Test that a particle in a harmonic trap oscillates about the center at the angular frequency `sqrt(k / m)`."""
    system = System(
        elements=[MassPointElement('ball', x=[2.0], m=2.0)],
        interactions=[HarmonicPotentialInteraction('trap', target='ball', k=8.0, center=1.0)],
        space=ContinuousSpace(n_dim=1),
    )
    trajectory = (
        Simulation(system, solver=ScipyODESolver(rtol=1e-10, atol=1e-12), dt=0.1)
        .check()
        .prepare()
        .run(until=3)
        .trajectory_
    )
    np.testing.assert_allclose(trajectory.values('ball.x')[:, 0], 1.0 + np.cos(2.0 * trajectory.times), atol=1e-6)
