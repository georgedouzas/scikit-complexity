"""Site of a sandpile, which holds a number of grains."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec


class SandpileCellElement(Element):
    """Site of a sandpile, which holds a number of grains.

    An `AvalancheInteraction` drops the grains and topples the sites that hold too many.

    Args:
        label:
            The label of the site or the lattice group.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        height:
            The initial number of grains.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.models.physics.criticality import SandpileCellElement
        >>> SandpileCellElement('site').height
        0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        height: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.height = height
