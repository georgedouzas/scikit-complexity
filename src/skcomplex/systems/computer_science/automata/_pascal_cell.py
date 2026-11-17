"""Cell of an automaton over the nonnegative integers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element
from ....core.specs import StateSpec


class PascalCellElement(Element):
    """Cell of an automaton over the nonnegative integers.

    Args:
        label:
            The label of the cell or the lattice group.

        value:
            The initial value, a nonnegative integer.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import PascalCellElement
        >>> PascalCellElement('cell', value=3).value
        3
    """

    def __init__(
        self,
        label: str,
        *,
        value: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label)
        self.value = value
