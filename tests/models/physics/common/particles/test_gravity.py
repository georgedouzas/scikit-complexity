"""Test the `_gravity` module of `skcomplex.models.physics.common.particles`."""

import numpy as np

from skcomplex.core.elements import ParticleElement
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models.physics.common.particles import GravityInteraction


def test_gravity_keeps_circular_orbit():
    """Test that a light body on a circular orbit keeps its radius."""
    system = System(
        elements=[
            ParticleElement('sun', m=1000.0),
            ParticleElement('planet', x=[10.0, 0.0], v=[0.0, 10.0], m=1e-6),
        ],
        interactions=[GravityInteraction('gravity', first='sun', second='planet', G=1.0)],
        space=ContinuousSpace(n_dim=2),
    )
    trajectory = (
        Simulation(system, solver=ScipyODESolver(rtol=1e-10, atol=1e-12), dt=0.5)
        .check()
        .prepare()
        .run(until=6)
        .trajectory_
    )
    radius = np.linalg.norm(trajectory.values('planet.x') - trajectory.values('sun.x'), axis=1)
    assert np.allclose(radius, 10.0, rtol=1e-4)
