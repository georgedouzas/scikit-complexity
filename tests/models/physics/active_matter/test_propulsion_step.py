"""Test the `_propulsion_step` module of `skcomplex.models.physics.active_matter`."""

import math
from typing import Annotated

import numpy as np
import numpy.typing as npt

from skcomplex.core.elements import Element
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.core.times import DiscreteTime
from skcomplex.engine.runtime import Simulation
from skcomplex.models.physics.active_matter import PropulsionStepInteraction
from skcomplex.specs import StateSpec


class WalkerElement(Element):
    """Walker with a position and a fixed heading."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, position=True)] = 0.0,
        theta: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.theta = theta


def test_position_steps_along_the_heading_every_step():
    """Test that a position steps the speed along a fixed heading every step of discrete time."""
    walker = WalkerElement('walker', x=[1.0, 1.0], theta=math.pi / 3)
    step = PropulsionStepInteraction('propulsion', target='walker', speed=0.5)
    system = System(elements=[walker], interactions=[step], space=ContinuousSpace(n_dim=2), time=DiscreteTime())
    trajectory = Simulation(system).check().prepare().run(until=4).trajectory_
    expected = [1.0 + 2.0 * math.cos(math.pi / 3), 1.0 + 2.0 * math.sin(math.pi / 3)]
    np.testing.assert_allclose(trajectory.values('walker.x')[-1], expected, atol=1e-12)
