"""Node of a Boolean network, a gene or protein that is on or off."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec


class BooleanNodeElement(Element):
    """Node of a Boolean network, a gene or protein that is on or off.

    Args:
        label:
            The label of the node.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value, 0 or 1.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import BooleanNodeElement
        >>> BooleanNodeElement('Sic1', x=1).x
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
