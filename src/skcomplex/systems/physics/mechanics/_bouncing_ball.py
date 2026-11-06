"""Ball at the height `h` above a floor, falling under the gravitational acceleration `g`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class BouncingBallElement(Element):
    """Ball at the height `h` above a floor, falling under the gravitational acceleration `g`.

    The height has order 2 and its first derivative `h_d1` is the vertical velocity.

    Args:
        label:
            The label of the ball.

        visual:
            How the ball is drawn, or `None` for none.

        h:
            The initial height.

        h_d1:
            The initial vertical velocity.

        g:
            The gravitational acceleration.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import BouncingBallElement
        >>> BouncingBallElement('ball', h=2.0).h
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        h: Annotated[float, StateSpec(order=2)] = 1.0,
        h_d1: float = 0.0,
        g: Annotated[float, ParameterSpec()] = 9.81,
    ) -> None:
        super().__init__(label, visual=visual)
        self.h = h
        self.h_d1 = h_d1
        self.g = g
