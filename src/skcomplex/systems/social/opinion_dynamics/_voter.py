"""Agent of the voter system, whose opinion the imitation sets to that of a random neighbour on a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec
from ._voter_view import VoterView


class VoterElement(Element):
    """Agent of the voter system, whose opinion the imitation sets to that of a random neighbour on a network.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, 0 or 1.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.opinion_dynamics import VoterElement
        >>> group = GroupElement('voters', element=VoterElement, size=20, values={'opinion': 1.0})
        >>> group.label, group.element.__name__, group.size
        ('voters', 'VoterElement', 20)
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
    def _get_view(cls) -> type[VoterView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return VoterView
