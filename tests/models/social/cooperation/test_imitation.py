"""Test the `_imitation` module of `skcomplex.models.social.cooperation`."""

from typing import Annotated

import numpy as np

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws import RuleLaw
from skcomplex.core.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.social.cooperation import imitate
from skcomplex.specs import PhaseSpec, PhasesScheduleSpec, StateSpec, VisualSpec


class ImitatorElement(Element):
    """Test helper Imitator."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        strategy: Annotated[float, StateSpec()] = 0.0,
        payoff: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.strategy = strategy
        self.payoff = payoff

    def _get_laws(self, s):
        """Test helper laws."""
        return [RuleLaw(self.copy, phase='imitation')]

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
        elements=[
            GroupElement(
                'players',
                element=ImitatorElement,
                size=width,
                values={'strategy': payoffs, 'payoff': payoffs},
                location=locations,
            ),
        ],
        space=LatticeSpace((width, 1), boundary='fixed', neighborhood='von_neumann'),
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('imitation', activation='simultaneous')])),
    )
    trajectory = Simulation(system, random_state=0).check().run(until=3).trajectory_
    for frame in range(4):
        players = trajectory.agents('players', frame)
        order = np.argsort(np.asarray(players.locations)[:, 0])
        assert players.values['strategy'][order].tolist() == [1.0] * (frame + 1) + [0.0] * (width - frame - 1)
