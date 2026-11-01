"""Value and exponent of the Pomeau-Manneville map `x' = x + x^z (mod 1)`, which a `MapInteraction` iterates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class IntermittencyMapElement(Element):
    """Value and exponent of the Pomeau-Manneville map `x' = x + x^z (mod 1)`, which a `MapInteraction` iterates.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value, between 0 and 1.

        z:
            The exponent of the tangency at 0, above 1.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import IntermittencyMapElement
        >>> IntermittencyMapElement('map', z=1.5).z
        1.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.01,
        z: Annotated[object, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.z = z
