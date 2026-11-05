"""Woman of the brothel movement system: a trafficking victim, or a woman who travels freely."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec
from ._traveller_view import TravellerView


class TravellerElement(Element):
    """Woman of the brothel movement system: a trafficking victim, or a woman who travels freely.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        victim:
            Whether the woman is a trafficking victim, 1, or not, 0.

        trips:
            The number of train trips between cities.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.organizations import TravellerElement
        >>> group = GroupElement('women', element=TravellerElement, size=4, values={'victim': 1.0})
        >>> group.label, group.element.__name__, group.size
        ('women', 'TravellerElement', 4)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'victim', 'color_range': (0.0, 1.0)}),
        *,
        victim: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        trips: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.victim = victim
        self.trips = trips

    @classmethod
    def _get_view(cls) -> type[TravellerView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return TravellerView
