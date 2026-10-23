"""Test the `_components` module of `skcomplex.core.formalisms.agents`."""

import pytest

from skcomplex.core.base import state
from skcomplex.core.components import Element
from skcomplex.core.formalisms.agents import Rule, memory
from skcomplex.core.runtime import Simulation
from skcomplex.core.system import System


class Saver(Element):
    """Test helper Saver."""

    remembers = {'money': 3}  # noqa: RUF012
    money: float = state(0.0)
    seen: float = state(0.0)

    def laws(self, s):
        """Test helper laws."""
        return [Rule(self.save)]

    def save(self, agent, model):
        """Test helper save."""
        agent.seen = float(len(memory(agent, 'money')))
        agent.money += 1


def test_memory_window():
    """Agents remember the last values of the states their type lists."""
    trajectory = Simulation(System(components=[Saver.group('savers', 2)]), random_state=0).run(until=5).trajectory_
    assert trajectory.agents('savers', 5).values['seen'].tolist() == [3.0, 3.0]


def test_memory_of_unremembered_state_raises():
    """Reading the memory of a state the type does not remember raises an error."""

    class Forgetful(Element):
        money: float = state(0.0)

        def laws(self, s):
            return [Rule(lambda agent, model: memory(agent, 'money'))]

    with pytest.raises(KeyError, match='remember'):
        Simulation(System(components=[Forgetful.group('agents', 1)])).run(until=1)
