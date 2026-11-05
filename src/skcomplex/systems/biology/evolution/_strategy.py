"""Strategy of an evolutionary game, with the share `x` of the population that plays it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class StrategyElement(Element):
    """Strategy of an evolutionary game, with the share `x` of the population that plays it.

    Args:
        label:
            The label of the strategy or of the group of strategies.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial share.

    Examples:
        >>> from skcomplex.systems.biology.evolution import StrategyElement
        >>> StrategyElement('hawks', x=0.4).x
        0.4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
