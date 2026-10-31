"""Test the `_damping` module of `skcomplex.core.interactions`."""

import numpy as np
import pytest

from skcomplex.core.interactions import DampingInteraction
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models.physics.common.particles import MassPointElement


def test_linear_drag_slows_particle():
    """Test that drag decays the velocity exponentially."""
    system = System(
        elements=[MassPointElement('p', x_d1=1.0)],
        interactions=[DampingInteraction('air', velocity='x_d1', target='p', gamma=0.5)],
        space=ContinuousSpace(n_dim=1),
    )
    velocity = Simulation(system, dt=0.5).check().prepare().run(until=2).trajectory_.values('p.x_d1')[-1, 0]
    assert velocity == pytest.approx(np.exp(-1.0), rel=1e-4)


def test_damping_decays_velocity_at_rate_gamma_over_mass():
    """Test that damping decays the velocity of a heavy particle as `exp(-gamma * t / m)`."""
    system = System(
        elements=[MassPointElement('p', x_d1=[2.0, -1.0], m=4.0)],
        interactions=[DampingInteraction('air', velocity='x_d1', target='p', gamma=2.0)],
        space=ContinuousSpace(n_dim=2),
    )
    trajectory = (
        Simulation(system, solver=ScipyODESolver(rtol=1e-10, atol=1e-12), dt=0.5)
        .check()
        .prepare()
        .run(until=4)
        .trajectory_
    )
    expected = np.exp(-0.5 * trajectory.times)[:, None] * np.array([2.0, -1.0])
    np.testing.assert_allclose(trajectory.values('p.x_d1'), expected, rtol=1e-7)
