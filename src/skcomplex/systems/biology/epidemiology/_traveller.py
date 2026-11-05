"""Resident of a city at a network node, who infects fellow residents and travels to neighbouring cities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._person import SUSCEPTIBLE
from ._traveller_view import TravellerView


class TravellerElement(Element):
    """Resident of a city at a network node, who infects fellow residents and travels to neighbouring cities.

    In its own activation, an infected traveller infects every susceptible traveller of its node with probability
    `tau`, by a `TransitionInteraction` whose members act on their cell, and another transition makes it recover after
    a residence of a number of steps. A `MovementInteraction` moves it to a random neighbouring node with probability
    `m`. Health is 0 susceptible, 2 infected, or 3 recovered.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        health:
            The health state.

        tau:
            The probability of infecting a susceptible traveller of the same node at every step.

        m:
            The probability of moving to a neighbouring node at every step.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.epidemiology import TravellerElement
        >>> GroupElement('travellers', element=TravellerElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'health', 'color_range': (0.0, 3.0)}),
        *,
        health: Annotated[float, StateSpec(domain=NN)] = SUSCEPTIBLE,
        tau: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        m: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.health = health
        self.tau = tau
        self.m = m

    @classmethod
    def _get_view(cls) -> type[TravellerView]:
        """Return the view of a traveller, `TravellerView`."""
        return TravellerView
