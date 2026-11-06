"""Link of a lattice with an SU(3) matrix, the color connection of lattice QCD."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import MatrixGroupSpec, StateSpec, VisualSpec

COLOR = MatrixGroupSpec('SU', 3)


class ColorLinkElement(Element):
    """Link of a lattice with an SU(3) matrix `U`, the parallel transport of color along the link.

    Args:
        label:
            The label of the link or the group on the links.

        visual:
            How the link is drawn, or `None` for `default_visual_`.

        U:
            The real components of the matrix of the link, the identity by default.

    Attributes:
        default_visual_:
            The visual of the link when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.waves import ColorLinkElement
        >>> len(ColorLinkElement('link').U)
        18
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        U: Annotated[object, StateSpec(group=COLOR)] = tuple(COLOR.identity),
    ) -> None:
        super().__init__(label, visual=visual)
        self.U = U
