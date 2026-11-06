"""Econophysics systems: wealth exchange between agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import TransferInteraction
from ....core.components.system import System
from ....core.specs import StateSpec, VisualSpec
from ..._catalog import register_system


class TraderElement(Element):
    """Agent of the Boltzmann wealth system, whose wealth in whole units changes only by gifts of one unit.

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


@register_system(
    field='economics/econophysics',
    tags=('wealth', 'agents', 'exchange'),
    references=('Dragulescu and Yakovenko 2000',),
    defaults={'until': 100},
)
def boltzmann_wealth(n_agents: int = 100) -> System:
    """Return a Boltzmann wealth exchange system where every agent starts with one unit.

    Every step, every agent with any wealth gives one unit to another agent drawn at random, in random order.

    Args:
        n_agents:
            The number of agents.

    Returns:
        system:
            The system, with the group `traders` and the transfer `gifts`.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import boltzmann_wealth
        >>> boltzmann_wealth(10).elements[0].prepare().size_
        10
    """
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
    return System(elements=[GroupElement('traders', element=TraderElement, size=n_agents)], interactions=[gifts])
