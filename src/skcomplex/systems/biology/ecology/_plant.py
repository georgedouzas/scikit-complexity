"""Plant that spreads into an empty neighbouring cell whenever it reaches maturity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._plant_view import PlantView


class PlantElement(Element):
    """Plant that spreads into an empty neighbouring cell whenever it reaches maturity.

    The system adds the ageing and the spreading as member interactions, and a plant that spreads starts its age anew.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        age:
            The steps since the plant last spread.

        maturity:
            The age at which the plant spreads.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.ecology import PlantElement
        >>> GroupElement('plants', element=PlantElement, size=100).size
        100
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'radius': 0.25}),
        *,
        age: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        maturity: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 4.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.age = age
        self.maturity = maturity

    @classmethod
    def _get_view(cls) -> type[PlantView]:
        """Return the view of a plant, `PlantView`."""
        return PlantView
