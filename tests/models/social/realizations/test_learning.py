"""Test the `_learning` module of `skcomplex.models.social.realizations`."""

import numpy as np

from skcomplex.core.base import state
from skcomplex.core.components import Element
from skcomplex.core.space import DiscreteTime, LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.formalisms.agents import Phase, Phases, Rule
from skcomplex.models.social.realizations import imitate


class Imitator(Element):
    """Test helper Imitator."""

    strategy: float = state(0.0)
    payoff: float = state(0.0)

    def laws(self, s):
        """Test helper laws."""
        return [Rule(self.copy, phase='imitation')]

    def copy(self, agent, model):
        """Test helper copy."""
        imitate(agent, agent.neighbors(), 'payoff', ('strategy', 'payoff'))


def test_simultaneous_imitation():
    """Every agent reads its neighbours' strategies from the start of a simultaneous phase."""
    width = 9
    locations = [(x, 0) for x in range(width)]
    payoffs = np.zeros(width)
    payoffs[0] = 1.0
    system = System(
        components=[Imitator.group('players', width, location=locations, strategy=payoffs, payoff=payoffs)],
        space=LatticeSpace((width, 1), boundary='fixed', neighborhood='von_neumann'),
        time=DiscreteTime(schedule=Phases([Phase('imitation', activation='simultaneous')])),
    )
    trajectory = Simulation(system, random_state=0).run(until=3).trajectory_
    for frame in range(4):
        players = trajectory.agents('players', frame)
        order = np.argsort(np.asarray(players.locations)[:, 0])
        assert players.values['strategy'][order].tolist() == [1.0] * (frame + 1) + [0.0] * (width - frame - 1)
