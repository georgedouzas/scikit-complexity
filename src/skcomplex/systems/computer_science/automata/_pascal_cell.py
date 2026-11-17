"""Cell of an automaton over the nonnegative integers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element
from ....core.components.spaces import BaseSpace
from ....core.types import Array


class PascalCellElement(Element):
    """Cell of an automaton over the nonnegative integers.

    With `space`, the element holds one cell on every site.

    Args:
        label:
            The label of the cells.

        value:
            The initial value, a nonnegative integer, or one value per cell.

        space:
            The space with one cell on every site, such as a lattice, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import PascalCellElement
        >>> PascalCellElement('cell', value=3).value
        3
    """

    def __init__(
        self,
        label: str,
        *,
        value: Annotated[float | Array, NN] = 0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.value = value
