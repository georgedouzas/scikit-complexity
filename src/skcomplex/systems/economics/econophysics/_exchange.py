"""Econophysics systems: wealth exchange between agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated, Protocol, cast

from sage.all__sagemath_symbolics import NN

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import PartnerSpec, UpdateLaw
from ....core.components.laws.utils import pairs, transfer
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System


class TraderView(AgentViewProtocol, Protocol):
    """Trader of the Boltzmann wealth system as its rule sees it, with its `wealth`."""

    wealth: float


class TraderElement(Element):
    """Agent of the Boltzmann wealth system.

    Its wealth in whole units changes only by gifts of one unit.

    Args:
        label:
            The label of the agent or group.

        wealth:
            The initial wealth.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import TraderElement
        >>> group = TraderElement('traders', size=100)
        >>> group.label, group.size
        ('traders', 100)
    """

    def __init__(
        self,
        label: str,
        *,
        wealth: Annotated[object, NN] = 1.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.wealth = wealth


class BoltzmannWealthSystem(BaseSystem):
    """A Boltzmann wealth exchange system where every agent starts with one unit.

    Every step, every agent with any wealth gives one unit to another agent drawn at random, in random order. The system
    has the group `traders` and the transfer `gifts`.

    Args:
        n_agents:
            The number of agents.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import BoltzmannWealthSystem
        >>> BoltzmannWealthSystem(10).prepare().system_.elements[0].size
        10
    """

    field = 'economics/econophysics'
    topics = ('wealth', 'agents', 'exchange')
    references = ('Dragulescu and Yakovenko 2000',)
    until = 100

    def __init__(
        self,
        n_agents: int = 100,
        *,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        gifts = Interaction('gifts', roles={'target': 'traders'}, laws=lambda s: [UpdateLaw(s, self._give)])
        self.system_ = System(
            self.label,
            elements=[TraderElement('traders', size=self.n_agents)],
            interactions=[gifts],
        )

    @staticmethod
    def _give(system: SystemViewProtocol) -> None:
        """Let every trader with wealth give one unit to a random trader, in random order."""
        traders = system.get_agents('target')
        found = cast(
            list[tuple[TraderView, TraderView]],
            list(pairs(traders, traders, system.rng, PartnerSpec('all', 'uniform'))),
        )
        for index in system.rng.permutation(len(found)):
            giver, receiver = found[int(index)]
            if giver.alive and receiver.alive and giver.wealth > 0:
                transfer(giver, receiver, 1.0, 'wealth')
