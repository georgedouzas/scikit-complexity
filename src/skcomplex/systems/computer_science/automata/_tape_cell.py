"""Cell of the tape of a Turing machine: a symbol, and the internal state of the head on it or -1."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, ZZ

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class TapeCellElement(Element):
    """Cell of the tape of a Turing machine: a symbol, and the internal state of the head on it or -1.

    The local rule of `turing_machine` lets the head on a cell write the symbol of its action table, move one cell left
    or right, or stay, and take its next internal state. A halted head keeps its cell and the tape stays unchanged.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        symbol:
            The initial symbol.

        head:
            The internal state of the head on the cell, from 0, or -1 for no head.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import TapeCellElement
        >>> TapeCellElement('cell').head
        -1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        symbol: Annotated[float, StateSpec(domain=NN)] = 0,
        head: Annotated[float, StateSpec(domain=ZZ)] = -1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.symbol = symbol
        self.head = head
