"""Test the `_damping` module of `skcomplex.systems.physics.common.particles`."""

import numpy as np
import pytest

from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.particles import DampingInteraction, MassPointElement


def test_linear_drag_slows_particle():
    """Test that drag decays the velocity exponentially."""
    system = System(
        elements=[MassPointElement('p', x=[0.0], values={"x'": [1.0]})],
        interactions=[DampingInteraction('air', target='p', gamma=0.5)],
        space=ContinuousSpace(n_dim=1),
    )
    velocity = system.prepare().simulate(until=2, dt=0.5).get_values("p.x'")[-1, 0]
    assert velocity == pytest.approx(np.exp(-1.0), rel=1e-4)


def test_damping_decays_velocity_at_rate_gamma_over_mass():
    """Test that damping decays the velocity of a heavy particle as `exp(-gamma * t / m)`."""
    system = System(
        elements=[MassPointElement('p', x=[0.0, 0.0], m=4.0, values={"x'": [2.0, -1.0]})],
        interactions=[DampingInteraction('air', target='p', gamma=2.0)],
        space=ContinuousSpace(n_dim=2),
        solver=ScipyODESolver(rtol=1e-10, atol=1e-12),
    )
    trajectory = system.prepare().simulate(until=4, dt=0.5)
    expected = np.exp(-0.5 * trajectory.times)[:, None] * np.array([2.0, -1.0])
    np.testing.assert_allclose(trajectory.get_values("p.x'"), expected, rtol=1e-7)
