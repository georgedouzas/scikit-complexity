"""Test the `_linear_drag` module of `skcomplex.systems.physics.common.particles`."""

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.particles import LinearDragInteraction


def test_linear_drag_decays_velocity_at_rate_gamma_over_mass():
    """Test that drag decays the velocity of a heavy ball in a plane as `exp(-gamma t / m)`."""
    ball = Element('ball', values={'x': 0.0, 'y': 0.0, 'm': 4.0})
    drag = LinearDragInteraction('air', target='ball', gamma=2.0, coordinates=('x', 'y'))
    solver = ScipyODESolver(rtol=1e-10, atol=1e-12)
    system = System(elements=[ball], interactions=[drag], solver=solver).prepare()
    trajectory = system.simulate(until=4.0, dt=0.5, initial_state={"ball.x'": 2.0, "ball.y'": -1.0})
    decay = np.exp(-0.5 * trajectory.times)
    np.testing.assert_allclose(trajectory.get_values("ball.x'"), 2.0 * decay, rtol=1e-7)
    np.testing.assert_allclose(trajectory.get_values("ball.y'"), -decay, rtol=1e-7)
