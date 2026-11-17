"""Node of a Boolean network, a gene or protein that is on or off."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from .....core.components.elements import Element
from .....core.specs import StateSpec


class BooleanNodeElement(Element):
    """Node of a Boolean network, a gene or protein that is on or off.

    Args:
        label:
            The label of the node.

        x:
            The initial value, 0 or 1.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import BooleanNodeElement
        >>> BooleanNodeElement('Sic1', x=1).x
        1
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
    ) -> None:
        super().__init__(label)
        self.x = x
