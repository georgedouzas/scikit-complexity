"""Test the `_self_propulsion` module of `skcomplex.systems.physics.common.particles`."""

import math

import numpy as np

from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.common.oscillators import SitedOscillatorElement
from skcomplex.systems.physics.common.particles import SelfPropulsionInteraction


def test_position_drifts_along_a_turning_heading():
    """Test that a position driven along a heading turning at `omega` traces a circle of radius `speed / omega`."""
    swimmer = SitedOscillatorElement('swimmer', x=[0.0, 0.0], theta=0.0, omega=1.0)
    propulsion = SelfPropulsionInteraction('propulsion', target='swimmer', speed=2.0)
    system = System(elements=[swimmer], interactions=[propulsion], space=ContinuousSpace(n_dim=2))
    trajectory = simulate(system, until=math.pi, dt=0.1)
    np.testing.assert_allclose(trajectory.values('swimmer.x')[-1], [0.0, 4.0], atol=1e-5)
