"""Test the `_periodic_force` module of `skcomplex.systems.physics.common.particles`."""

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.particles import PeriodicForceInteraction


def test_periodic_force_moves_a_free_mass_as_a_cosine():
    """Test that a free mass at rest under `F cos(omega t)` follows `F (1 - cos(omega t)) / (m omega²)`."""
    ball = Element('ball', values={'x': 0.0, 'm': 2.0})
    force = PeriodicForceInteraction('drive', target='ball', F=3.0, omega=2.0)
    solver = ScipyODESolver(rtol=1e-10, atol=1e-12)
    trajectory = System(elements=[ball], interactions=[force], solver=solver).prepare().simulate(until=3.0, dt=0.1)
    expected = 3.0 * (1 - np.cos(2.0 * trajectory.times)) / (2.0 * 4.0)
    np.testing.assert_allclose(trajectory.get_values('ball.x'), expected, atol=1e-8)
