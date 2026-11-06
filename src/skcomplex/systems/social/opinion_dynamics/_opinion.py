"""Node of a consensus process with a continuous opinion."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class OpinionElement(Element):
    """Node of a consensus process with a continuous opinion.

    Args:
        label:
            The label of the node or the group.

        visual:
            How the node is drawn, or `None` for `default_visual_`.

        x:
            The initial opinion.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import OpinionElement
        >>> OpinionElement('node', x=0.5).x
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
