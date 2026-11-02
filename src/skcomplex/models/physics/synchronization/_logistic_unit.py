"""Value `x` in `[0, 1]` of a logistic map, which maps of the coupling iterate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec


class LogisticUnitElement(Element):
    """Value `x` in `[0, 1]` of a logistic map, which maps of the coupling iterate.

    Args:
        label:
            The label of the unit.

        visual:
            How the unit is drawn, or `None` for `default_visual_`.

        x:
            The initial value.

    Attributes:
        default_visual_:
            The visual of the unit when it sets none.

    Examples:
        >>> from skcomplex.models.physics.synchronization import LogisticUnitElement
        >>> LogisticUnitElement('unit').x
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
