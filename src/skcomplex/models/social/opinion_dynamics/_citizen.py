"""Agent of the Galam majority model, who holds a binary opinion."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec
from ._citizen_view import CitizenView


class CitizenElement(Element):
    """Agent of the Galam majority model, who holds a binary opinion.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, 0 or 1.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.social.opinion_dynamics import CitizenElement
        >>> group = GroupElement('citizens', element=CitizenElement, size=10, values={'opinion': 1.0})
        >>> group.label, group.element.__name__, group.size
        ('citizens', 'CitizenElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        opinion: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion

    @classmethod
    def _get_view(cls) -> type[CitizenView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return CitizenView
