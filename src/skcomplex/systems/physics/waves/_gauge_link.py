"""Link of a lattice with an angle, the U(1) or Z_N variable of a lattice gauge theory."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import pi

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class GaugeLinkElement(Element):
    """Link of a lattice with an angle `theta`, the U(1) or Z_N variable of a lattice gauge theory.

    Args:
        label:
            The label of the link or the group on the links.

        visual:
            How the link is drawn, or `None` for `default_visual_`.

        theta:
            The angle of the link.

    Attributes:
        default_visual_:
            The visual of the link when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.waves import GaugeLinkElement
        >>> GaugeLinkElement('link', theta=0.5).theta
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
