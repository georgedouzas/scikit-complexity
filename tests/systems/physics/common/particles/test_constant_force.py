"""Test the `_constant_force` module of `skcomplex.systems.physics.common.particles`."""

import numpy as np

from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.particles import ConstantForceInteraction, MassPointElement


def test_particle_constant_acceleration():
    """Test that a constant force displaces a particle by `F * t**2 / (2 * m)`."""
    system = System(
        elements=[MassPointElement('p', x=[0.0, 0.0, 0.0], m=2.0)],
        interactions=[ConstantForceInteraction('push', target='p', F=[4.0, 0.0, -2.0])],
        space=ContinuousSpace(n_dim=3),
    )
    position = system.prepare().simulate(until=2, dt=0.5).get_values('p.x')[-1]
    assert np.allclose(position, [4.0, 0.0, -2.0])


def test_a_number_pushes_every_component():
    """Test that a force given as a number pushes every component of the position."""
    system = System(
        elements=[MassPointElement('p', x=[0.0, 0.0])],
        interactions=[ConstantForceInteraction('push', target='p', F=1.0)],
        space=ContinuousSpace(n_dim=2),
    )
    assert np.allclose(system.prepare().simulate(until=2, dt=1.0).get_values('p.x')[-1], [2.0, 2.0])
