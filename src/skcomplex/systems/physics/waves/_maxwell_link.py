"""Link of a lattice with a vector potential whose velocity is minus the electric field."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class MaxwellLinkElement(Element):
    """Link of a lattice with a vector potential `A` of second order, whose velocity `A_d1` is the field `-E`.

    Args:
        label:
            The label of the link or the group on the links.

        visual:
            How the link is drawn, or `None` for `default_visual_`.

        A:
            The initial vector potential.

        A_d1:
            The initial velocity of the vector potential, minus the electric field.

    Attributes:
        default_visual_:
            The visual of the link when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.waves import MaxwellLinkElement
        >>> MaxwellLinkElement('link', A=0.5).A
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec(order=2)] = 0.0,
        A_d1: float = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.A_d1 = A_d1
