"""Define binary cells, the cells of automata whose state is dead or alive."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.components.spaces import BaseSpace
from ....core.types import Array


class BinaryCellElement(Element):
    """Cell of an automaton whose value `alive` is 0 or 1.

    The local rule of an interaction changes the value. With `space`, the element holds one cell on every site.

    Args:
        label:
            The label of the cells.

        alive:
            The initial value, 0 or 1, or one value per cell.

        space:
            The space with one cell on every site, such as a lattice, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import BinaryCellElement
        >>> BinaryCellElement('cell', alive=1).alive
        1
    """

    def __init__(
        self,
        label: str,
        *,
        alive: Annotated[float | Array, {0, 1}] = 0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.alive = alive
