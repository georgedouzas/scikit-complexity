"""Define the cell of the golden mean shift, whose configurations never hold two adjacent ones."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....core.spaces import LatticeNeighborhood
from ....specs import StateSpec, SubshiftSpec, VisualSpec

GOLDEN_MEAN = SubshiftSpec(LatticeNeighborhood(((0,), (1,))), ((1, 1),))


class GoldenCellElement(Element):
    """Cell of the golden mean shift, whose state `alive` is 0 or 1 and whose configurations hold no two adjacent ones.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        alive:
            The initial value, 0 or 1.

    Examples:
        >>> from skcomplex.models.computer_science.automata import GoldenCellElement
        >>> GoldenCellElement('cell', alive=1).alive
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        alive: Annotated[float, StateSpec(domain=GOLDEN_MEAN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.alive = alive
