"""Speed and wall phase of the Fermi-Ulam map of a ball bouncing between a fixed and an oscillating wall."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class FermiUlamMapElement(Element):
    """Speed and wall phase of the Fermi-Ulam map of a ball bouncing between a fixed and an oscillating wall.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial speed of the ball.

        theta:
            The initial phase of the oscillating wall.

        epsilon:
            The amplitude of the speed the oscillating wall gives the ball.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import FermiUlamMapElement
        >>> FermiUlamMapElement('map', epsilon=0.2).epsilon
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec()] = 0.5,
        theta: Annotated[object, StateSpec()] = 1.0,
        epsilon: Annotated[object, ParameterSpec()] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.theta = theta
        self.epsilon = epsilon
