"""Test the `_constant_force` module of `skcomplex.models.physics.common.particles`."""

import numpy as np

from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.physics.common.particles import ConstantForceInteraction, MassPointElement


def test_particle_constant_acceleration():
    """Test that a constant force displaces a particle by `F * t**2 / (2 * m)`."""
    system = System(
        elements=[MassPointElement('p', m=2.0)],
        interactions=[ConstantForceInteraction('push', target='p', F=[4.0, 0.0, -2.0])],
        space=ContinuousSpace(n_dim=3),
    )
    position = Simulation(system, dt=0.5).check().prepare().run(until=2).trajectory_.values('p.x')[-1]
    assert np.allclose(position, [4.0, 0.0, -2.0])
