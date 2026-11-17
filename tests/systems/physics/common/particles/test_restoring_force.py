"""Test the `_restoring_force` module of `skcomplex.systems.physics.common.particles`."""

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.particles import RestoringForceInteraction


def test_restoring_force_oscillates_about_its_center():
    """Test that a mass on a spring oscillates about the center at the angular frequency `sqrt(k / m)`."""
    ball = Element('ball', values={'x': 2.0, 'm': 2.0})
    trap = RestoringForceInteraction('trap', target='ball', k=8.0, center=1.0)
    solver = ScipyODESolver(rtol=1e-10, atol=1e-12)
    trajectory = System(elements=[ball], interactions=[trap], solver=solver).prepare().simulate(until=3.0, dt=0.1)
    np.testing.assert_allclose(trajectory.get_values('ball.x'), 1.0 + np.cos(2.0 * trajectory.times), atol=1e-6)
