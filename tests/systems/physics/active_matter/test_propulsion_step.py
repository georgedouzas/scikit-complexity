"""Test the `_propulsion_step` module of `skcomplex.systems.physics.active_matter`."""

import math
from typing import Annotated

import numpy as np
import numpy.typing as npt

from skcomplex.core.components.elements import Element
from skcomplex.core.components.spaces import ContinuousSpace, DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec
from skcomplex.systems.physics.active_matter import PropulsionStepInteraction


class WalkerElement(Element):
    """Walker with a position and a fixed heading."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        theta: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.theta = theta


def test_position_steps_along_the_heading_every_step():
    """Test that a position steps the speed along a fixed heading every step of discrete time."""
    walker = WalkerElement('walker', x=[1.0, 1.0], theta=math.pi / 3)
    step = PropulsionStepInteraction('propulsion', target='walker', speed=0.5)
    system = CompositeSystem(
        elements=[walker],
        interactions=[step],
        space=ContinuousSpace(n_dim=2),
        time=DiscreteTime(),
    )
    trajectory = simulate(system, until=4)
    expected = [1.0 + 2.0 * math.cos(math.pi / 3), 1.0 + 2.0 * math.sin(math.pi / 3)]
    np.testing.assert_allclose(trajectory.values('walker.x')[-1], expected, atol=1e-12)
