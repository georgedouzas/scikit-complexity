"""Define the element that holds the integral of a signal, whose rate a controller sets."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class IntegratorElement(Element):
    """Element that holds the integral of a signal, whose rate a controller sets.

    Args:
        label:
            The label of the integrator.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        value:
            The initial value of the integral.

    Examples:
        >>> from skcomplex.systems.engineering.control import IntegratorElement
        >>> IntegratorElement('memory').value
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        value: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.value = value
