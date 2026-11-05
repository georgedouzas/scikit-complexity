"""Agent of the Sznajd system, who convinces the neighbours of an agreeing pair."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec
from ._persuader_view import PersuaderView


class PersuaderElement(Element):
    """Agent of the Sznajd system, who convinces the neighbours of an agreeing pair.

    A step of an agent picks a random neighbour, and when both share an opinion every neighbour of either adopts it.
    On a ring this is the original one-dimensional rule, where the pair convinces the two agents beside it.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, 0 or 1.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.opinion_dynamics import PersuaderElement
        >>> group = GroupElement('people', element=PersuaderElement, size=10, values={'opinion': 1.0})
        >>> group.label, group.element.__name__, group.size
        ('people', 'PersuaderElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (0.0, 1.0)}),
        *,
        opinion: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion

    @classmethod
    def _get_view(cls) -> type[PersuaderView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return PersuaderView
