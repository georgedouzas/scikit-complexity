"""Cell of an automaton over the nonnegative integers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec


class PascalCellElement(Element):
    """Cell of an automaton over the nonnegative integers.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        value:
            The initial value, a nonnegative integer.

    Examples:
        >>> from skcomplex.models.computer_science.automata import PascalCellElement
        >>> PascalCellElement('cell', value=3).value
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        value: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.value = value
