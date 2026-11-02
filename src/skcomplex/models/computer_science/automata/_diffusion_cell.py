"""Cell of a lattice with a content and the share of the difference it exchanges with each neighbour in a step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class DiffusionCellElement(Element):
    """Cell of a lattice with a content and the share of the difference it exchanges with each neighbour in a step.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        level:
            The initial content.

        r:
            The share of the difference with a neighbour exchanged in a step, below `1 / (2 d)` in `d` dimensions.

    Examples:
        >>> from skcomplex.models.computer_science.automata import DiffusionCellElement
        >>> DiffusionCellElement('cell', r=0.2).r
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        level: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.level = level
        self.r = r
