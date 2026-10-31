"""Test the `_spring` module of `skcomplex.models.physics.common.particles`."""

import numpy as np

from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models.physics.common.particles import MassPointElement, SpringInteraction


def test_spring_conserves_energy():
    """Test that two particles joined by a spring keep their total energy."""
    system = System(
        elements=[MassPointElement('a', x=[0.0, 0.0], x_d1=[0.0, 1.0]), MassPointElement('b', x=[2.0, 0.0], m=2.0)],
        interactions=[SpringInteraction('spring', velocity='x_d1', first='a', second='b', k=3.0, length=1.0)],
        space=ContinuousSpace(n_dim=2),
    )
    trajectory = (
        Simulation(system, solver=ScipyODESolver(rtol=1e-10, atol=1e-12), dt=0.1)
        .check()
        .prepare()
        .run(until=5)
        .trajectory_
    )
    kinetic = 0.5 * (trajectory.values('a.x_d1') ** 2).sum(axis=1) + 0.5 * 2.0 * (trajectory.values('b.x_d1') ** 2).sum(
        axis=1,
    )
    distance = np.linalg.norm(trajectory.values('a.x') - trajectory.values('b.x'), axis=1)
    energy = kinetic + 0.5 * 3.0 * (distance - 1.0) ** 2
    assert np.allclose(energy, energy[0], rtol=1e-6)
