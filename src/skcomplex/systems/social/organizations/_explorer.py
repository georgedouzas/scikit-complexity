"""Individual of the mobility systems, with its position, its number of distinct locations, and its last level."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec
from ._explorer_view import ExplorerView


class ExplorerElement(Element):
    """Individual of the mobility systems, with its position, its number of distinct locations, and its last level.

    Args:
        label:
            The label of the individual or the group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        x:
            The first coordinate of the current location.

        y:
            The second coordinate of the current location.

        distinct:
            The number of distinct locations visited.

        level:
            The level-distance of the last displacement in a hierarchy of containers, 0 without one.

    Examples:
        >>> from skcomplex.systems.social.organizations import ExplorerElement
        >>> ExplorerElement('individual').distinct
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.0,
        distinct: Annotated[float, StateSpec(domain=NN)] = 1.0,
        level: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.distinct = distinct
        self.level = level

    @classmethod
    def _get_view(cls) -> type[ExplorerView]:
        """Return the class of the views of the agents, whose attributes are the states."""
        return ExplorerView
