"""Test the `_spring` module of `skcomplex.systems.physics.common.particles`."""

import numpy as np

from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.particles import MassPointElement, SpringInteraction


def test_spring_conserves_energy():
    """Test that two particles joined by a spring keep their total energy."""
    system = System(
        elements=[
            MassPointElement('a', x=[0.0, 0.0], values={"x'": [0.0, 1.0]}),
            MassPointElement('b', x=[2.0, 0.0], m=2.0),
        ],
        interactions=[SpringInteraction('spring', first='a', second='b', k=3.0, length=1.0)],
        space=ContinuousSpace(n_dim=2),
        solver=ScipyODESolver(rtol=1e-10, atol=1e-12),
    )
    trajectory = system.prepare().simulate(until=5, dt=0.1)
    speeds = [(trajectory.get_values(f"{name}.x'") ** 2).sum(axis=1) for name in 'ab']
    kinetic = 0.5 * speeds[0] + 0.5 * 2.0 * speeds[1]
    distance = np.linalg.norm(trajectory.get_values('a.x') - trajectory.get_values('b.x'), axis=1)
    energy = kinetic + 0.5 * 3.0 * (distance - 1.0) ** 2
    assert np.allclose(energy, energy[0], rtol=1e-6)
