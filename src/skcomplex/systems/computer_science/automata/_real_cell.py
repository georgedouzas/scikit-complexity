"""Cell of an automaton over the real numbers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import RR

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class RealCellElement(Element):
    """Cell of an automaton over the real numbers.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        value:
            The initial value.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import RealCellElement
        >>> RealCellElement('cell', value=0.5).value
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        value: Annotated[float, StateSpec(domain=RR)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.value = value
