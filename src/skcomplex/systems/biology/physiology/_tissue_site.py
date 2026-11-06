"""Site of a lattice tissue, which holds the label of the cell that covers it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class TissueSiteElement(Element):
    """Site of a lattice tissue, which holds the label of the cell that covers it.

    The sites of one label form a cell of the cellular Potts model, and `CellAdhesionInteraction` moves the boundaries
    between cells by copying the label of a neighbouring site.

    Args:
        label:
            The label of the site or the lattice group.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        cell:
            The initial label of the cell that covers the site, from 0.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.systems.biology.physiology import TissueSiteElement
        >>> TissueSiteElement('site', cell=3).cell
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        cell: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.cell = cell
