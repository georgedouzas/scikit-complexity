"""Cell holding a digit of a q-adic integer, from 0 to `base - 1`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class DigitCellElement(Element):
    """Cell holding a digit of a q-adic integer, from 0 to `base - 1`, the label of a vertex of the q-ary tree.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        digit:
            The initial digit.

        base:
            The base `q` of the digits, at least 2.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import DigitCellElement
        >>> DigitCellElement('cell', digit=1).base
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        digit: Annotated[float, StateSpec(domain=ZZ)] = 0,
        base: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.digit = digit
        self.base = base
