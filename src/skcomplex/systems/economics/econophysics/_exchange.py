"""Econophysics systems: wealth exchange between agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import TransferInteraction
from ....core.components.systems import BaseSystem, System
from ....core.specs import StateSpec, VisualSpec


class TraderElement(Element):
    """Agent of the Boltzmann wealth system.

    Its wealth in whole units changes only by gifts of one unit.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        wealth:
            The initial wealth.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import TraderElement
        >>> group = GroupElement('traders', element=TraderElement, size=100)
        >>> group.label, group.element.__name__, group.size
        ('traders', 'TraderElement', 100)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        wealth: Annotated[float, StateSpec(domain=NN)] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
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

    Examples:
        >>> from skcomplex.systems.economics.econophysics import BoltzmannWealthSystem
        >>> BoltzmannWealthSystem(10).prepare().system_.elements[0].prepare().size_
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
    ) -> None:
        super().__init__(label)
        self.n_agents = n_agents

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        gifts = TransferInteraction(
            'gifts',
            source='traders',
            target='traders',
            state='wealth',
            amount=1.0,
            pairing='random',
            guard=lambda giver, receiver, system: giver.wealth > 0,
            form='agent',
        )
        return System(
            self.label,
            elements=[GroupElement('traders', element=TraderElement, size=self.n_agents)],
            interactions=[gifts],
        )
