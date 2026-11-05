"""Test the `_regrowth` module of `skcomplex.models.economics.common.resources`."""

from typing import Annotated

import numpy as np

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.spaces import DiscreteTime
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.economics.common.resources import RegrowthInteraction
from skcomplex.specs import ParameterSpec, StateSpec
from skcomplex.types import AnyArray


class StockElement(Element):
    """Stock `level` of a resource that grows back by `growth` a step up to `capacity`."""

    def __init__(
        self,
        label: str,
        level: Annotated[float, StateSpec()] = 0.0,
        capacity: Annotated[float, StateSpec()] = 4.0,
        growth: Annotated[float, ParameterSpec()] = 1.5,
    ) -> None:
        super().__init__(label)
        self.level = level
        self.capacity = capacity
        self.growth = growth


def _levels(*, relative: bool) -> AnyArray:
    """Return the levels of two stocks over three steps of regrowth, one row per frame."""
    stocks = GroupElement('stocks', element=StockElement, size=2, values={'level': [0.0, 3.0]})
    regrowth = RegrowthInteraction('regrowth', target='stocks', state='level', growth='growth', relative=relative)
    system = System(elements=[stocks], interactions=[regrowth], time=DiscreteTime())
    trajectory = Simulation(system, random_state=0).prepare().run(until=3).trajectory_
    return np.array([trajectory.agents('stocks', frame).values['level'] for frame in range(4)])


def test_regrowth_adds_an_amount_up_to_the_capacity():
    """Test that an absolute regrowth adds its amount every step and stops at the capacity."""
    levels = _levels(relative=False)
    np.testing.assert_allclose(levels[:, 0], [0.0, 1.5, 3.0, 4.0])
    np.testing.assert_allclose(levels[:, 1], [3.0, 4.0, 4.0, 4.0])


def test_relative_regrowth_adds_a_share_of_the_capacity():
    """Test that a relative regrowth adds its share of the capacity every step."""
    np.testing.assert_allclose(_levels(relative=True)[:, 0], [0.0, 4.0, 4.0, 4.0])
