"""Link of a lattice with an SU(2) matrix, the isospin connection of a lattice gauge theory."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import MatrixGroupSpec, StateSpec, VisualSpec

ISOSPIN = MatrixGroupSpec('SU', 2)


class IsospinLinkElement(Element):
    """Link of a lattice with an SU(2) matrix `U`, the parallel transport of isospin along the link.

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
        >>> from skcomplex.systems.physics.waves import IsospinLinkElement
        >>> len(IsospinLinkElement('link').U)
        8
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        U: Annotated[object, StateSpec(group=ISOSPIN)] = tuple(ISOSPIN.identity),
    ) -> None:
        super().__init__(label, visual=visual)
        self.U = U
