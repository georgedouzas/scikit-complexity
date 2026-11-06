"""Value and growth rate of the logistic map `x -> r x (1 - x)`, which a `MapInteraction` iterates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class LogisticMapElement(Element):
    """Value and growth rate of the logistic map `x -> r x (1 - x)`, which a `MapInteraction` iterates.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value, between 0 and 1.

        r:
            The growth rate, between 0 and 4.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import LogisticMapElement
        >>> LogisticMapElement('map', r=3.2).r
        3.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.2,
        r: Annotated[object, ParameterSpec()] = 3.7,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.r = r
