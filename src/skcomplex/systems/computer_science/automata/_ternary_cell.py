"""Cell of a subshift over the alphabet `{0, 1, 2}`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class TernaryCellElement(Element):
    """Cell of a subshift over the alphabet `{0, 1, 2}`, such as the context-free subshift.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        symbol:
            The initial symbol, 0, 1, or 2.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import TernaryCellElement
        >>> TernaryCellElement('cell', symbol=2).symbol
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        symbol: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1, 2]))] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.symbol = symbol
