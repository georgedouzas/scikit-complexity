"""Test the `_memory` module of `skcomplex.engine.solvers`."""

from typing import Annotated

import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws import RuleLaw
from skcomplex.core.specs import StateSpec, VisualSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import memory


class SaverElement(Element):
    """Test helper Saver."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        money: Annotated[float, StateSpec(lags=3)] = 0.0,
        seen: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.money = money
        self.seen = seen

    def _get_laws(self, s):
        """Test helper laws."""
        return [RuleLaw(self.save)]

    def save(self, agent, model):
        """Test helper save."""
        agent.seen = float(len(memory(agent, 'money')))
        agent.money += 1


def test_memory_window():
    """Agents remember the last values of the states their type lists."""
    trajectory = (
        Simulation(System(elements=[GroupElement('savers', element=SaverElement, size=2)]), random_state=0)
        .run(until=5)
        .trajectory_
    )
    assert trajectory.agents('savers', 5).values['seen'].tolist() == [3.0, 3.0]


def test_memory_of_unremembered_state_raises():
    """Reading the memory of a state the type does not remember raises an error."""

    class ForgetfulElement(Element):
        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            money: Annotated[float, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.money = money

        def _get_laws(self, s):
            return [RuleLaw(lambda agent, model: memory(agent, 'money'))]

    with pytest.raises(KeyError, match='remember'):
        Simulation(System(elements=[GroupElement('agents', element=ForgetfulElement, size=1)])).run(until=1)
