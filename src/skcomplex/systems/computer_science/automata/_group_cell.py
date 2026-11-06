"""Cell of an automaton whose alphabet is a group, holding a group element by its number."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import ZZ

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class GroupCellElement(Element):
    """Cell of an automaton whose alphabet is a group, holding a group element by its number.

    The number is the site of the element in a group space, or its position in the list of a finite group, and a
    negative number names no element, the fill value read outside a ball.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        value:
            The number of the initial group element, 0 for the identity.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import GroupCellElement
        >>> GroupCellElement('cell', value=2).value
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        value: Annotated[float, StateSpec(domain=ZZ)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.value = value
