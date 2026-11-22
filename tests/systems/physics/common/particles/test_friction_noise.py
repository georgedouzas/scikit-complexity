"""Test the `_friction_noise` module of `skcomplex.systems.physics.common.particles`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.particles import FrictionNoiseInteraction


def test_friction_noise_diffuses_every_axis_with_temperature_over_friction():
    """Test that beads in a plane spread with the variance `2 temperature t / gamma` along each axis."""
    beads = Element('beads', values={'x': np.zeros((4000, 2)), 'gamma': 4.0}, size=4000)
    bath = FrictionNoiseInteraction('bath', target='beads', temperature=2.0)
    solver = EulerMaruyamaSolver(dt=0.05)
    system = System(elements=[beads], interactions=[bath], space=ContinuousSpace(n_dim=2), solver=solver).prepare()
    positions = system.simulate(until=5.0, dt=5.0, random_state=0).get_values('beads.x')[-1]
    assert positions.var(axis=0) == pytest.approx([5.0, 5.0], rel=0.1)
