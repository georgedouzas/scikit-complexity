"""Real and imaginary parts of Newton's method for `z^3 = 1`, `z' = (2 z^3 + 1) / (3 z^2)`, which maps iterate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class NewtonMapElement(Element):
    """Real and imaginary parts of Newton's method for `z^3 = 1`, `z' = (2 z^3 + 1) / (3 z^2)`, which maps iterate.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial real part.

        y:
            The initial imaginary part.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import NewtonMapElement
        >>> NewtonMapElement('map', x=1.5).x
        1.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.5,
        y: Annotated[object, StateSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
