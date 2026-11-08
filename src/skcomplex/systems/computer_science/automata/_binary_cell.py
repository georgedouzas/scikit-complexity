"""Define binary cells, the cells of automata whose state is dead or alive."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class BinaryCellElement(Element):
    """Cell of an automaton whose state `alive` is 0 or 1.

    The local rule of an interaction changes the state.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        alive:
            The initial value, 0 or 1.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import BinaryCellElement
        >>> BinaryCellElement('cell', alive=1).alive
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.alive = alive
