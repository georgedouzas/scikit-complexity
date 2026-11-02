"""Self-propelled point particle with a heading `theta`, in a periodic square."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec
from ._flocker_view import FlockerView


class FlockerElement(Element):
    """Self-propelled point particle with a heading `theta`, in a periodic square.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        theta:
            The initial heading.

    Examples:
        >>> from skcomplex.models.physics.active_matter import FlockerElement
        >>> FlockerElement('bird', theta=1.0).theta
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'theta', 'color_range': (0.0, 2 * math.pi)}),
        *,
        theta: Annotated[float, StateSpec(role='heading')] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta

    @classmethod
    def _get_view(cls) -> type:
        """Return the class of the views of the element's agents."""
        return FlockerView
