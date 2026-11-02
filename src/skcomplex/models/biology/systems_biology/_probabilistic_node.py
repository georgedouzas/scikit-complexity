"""Node of a probabilistic Boolean network, on or off, with the index of the function that set it last."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec


class ProbabilisticNodeElement(Element):
    """Node of a probabilistic Boolean network, on or off, with the index of the function that set it last.

    Args:
        label:
            The label of the node.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value, 0 or 1.

        function:
            The index of the candidate function of the last update, 0 at the start.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import ProbabilisticNodeElement
        >>> ProbabilisticNodeElement('x1', x=1).x
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        function: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.function = function
