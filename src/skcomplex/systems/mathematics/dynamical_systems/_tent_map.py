"""Value and slope of the tent map `x' = mu min(x, 1 - x)`, which a `MapInteraction` iterates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class TentMapElement(Element):
    """Value and slope of the tent map `x' = mu min(x, 1 - x)`, which a `MapInteraction` iterates.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value, between 0 and 1.

        mu:
            The slope, between 0 and 2.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import TentMapElement
        >>> TentMapElement('map', mu=1.5).mu
        1.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.3,
        mu: Annotated[object, ParameterSpec()] = 1.99,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.mu = mu
