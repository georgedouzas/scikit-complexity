"""Source of symbols `0` to `n - 1`, whose symbol a transition matrix steps in discrete time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class MarkovSourceElement(Element):
    """Source of symbols `0` to `n - 1`, whose symbol a transition matrix steps in discrete time.

    Args:
        label:
            The label of the source.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        symbol:
            The initial symbol.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import MarkovSourceElement
        >>> MarkovSourceElement('source', symbol=1).symbol
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        symbol: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.symbol = symbol
