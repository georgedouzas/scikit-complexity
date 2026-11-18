"""Test the `_propulsion_step` module of `skcomplex.systems.physics.common.particles`."""

import math

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.systems import System
from skcomplex.systems.physics.active_matter import PropulsionStepInteraction


def test_position_steps_along_the_heading_every_step():
    """Test that a position steps the speed along a fixed heading every step of discrete time."""
    walker = Element('walker', values={'x': np.array([1.0, 1.0]), 'theta': math.pi / 3})
    step = PropulsionStepInteraction('propulsion', target='walker', speed=0.5)
    system = System(elements=[walker], interactions=[step], space=ContinuousSpace(n_dim=2)).prepare()
    trajectory = system.simulate(until=4)
    expected = [1.0 + 2.0 * math.cos(math.pi / 3), 1.0 + 2.0 * math.sin(math.pi / 3)]
    np.testing.assert_allclose(trajectory.get_values('walker.x')[-1], expected, atol=1e-12)
