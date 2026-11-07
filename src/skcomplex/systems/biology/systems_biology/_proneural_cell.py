"""Cell of an epithelium that adopts the neural fate when the inhibition from its neighbours is weak."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ProneuralCellElement(Element):
    """Cell of an epithelium that adopts the neural fate when the inhibition from its neighbours is weak.

    The state `a` runs from 0, the epidermal fate, to 1, the neural fate of a sensory bristle. The sigmoids of the
    lateral inhibition share the width `width`.

    Args:
        label:
            The label of the cell or of the group of cells.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        a:
            The initial neural state.

        width:
            The width of the sigmoids.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import ProneuralCellElement
        >>> ProneuralCellElement('cell', a=0.1).a
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.0,
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.127,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a
        self.width = width
