"""Test the `_components` module of `skcomplex.formalisms.agents`."""

import pytest
from skcomplex.core.components import Element

from skcomplex.core.base import state
from skcomplex.core.elements import GroupElement
from skcomplex.core.laws import RuleLaw
from skcomplex.core.system import System
from skcomplex.engine.backends import memory
from skcomplex.engine.runtime import Simulation


class SaverElement(Element):
    """Test helper Saver."""

    remembers = {'money': 3}  # noqa: RUF012
    money: float = state(0.0)
    seen: float = state(0.0)

    def laws(self, s):
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
        money: float = state(0.0)

        def laws(self, s):
            return [RuleLaw(lambda agent, model: memory(agent, 'money'))]

    with pytest.raises(KeyError, match='remember'):
        Simulation(System(elements=[GroupElement('agents', element=ForgetfulElement, size=1)])).run(until=1)
