"""Cell of an automaton over the product `H x H` of the integers modulo `n` with itself."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class PairCellElement(Element):
    """Cell of an automaton over the product `H x H` of the integers modulo `n` with itself, or `K^2` for a prime `n`.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        first:
            The initial first coordinate, an integer from 0 to `modulus - 1`.

        second:
            The initial second coordinate, an integer from 0 to `modulus - 1`.

        modulus:
            The order `n` of the cyclic group `H`, the size of the field `K` when prime.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import PairCellElement
        >>> PairCellElement('cell', first=1).modulus
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        first: Annotated[float, StateSpec(domain=ZZ)] = 0,
        second: Annotated[float, StateSpec(domain=ZZ)] = 0,
        modulus: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.first = first
        self.second = second
        self.modulus = modulus
