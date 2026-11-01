"""Ball hanging on a string of length `length`, which swings with small amplitude about its rest position."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class HangingBallElement(Element):
    """Ball hanging on a string of length `length`, which swings with small amplitude about its rest position.

    The horizontal position `x` has order 2, and its first derivative `x_d1` is the velocity. For small amplitudes
    the ball moves in the potential `g (x - rest)² / (2 length)`.

    Args:
        label:
            The label of the ball.

        visual:
            How the ball is drawn, or `None` for none.

        x:
            The initial horizontal position.

        x_d1:
            The initial horizontal velocity.

        rest:
            The rest position.

        g:
            The gravitational acceleration.

        length:
            The length of the string.

    Examples:
        >>> from skcomplex.models.physics.mechanics import HangingBallElement
        >>> HangingBallElement('ball', rest=0.5).rest
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        rest: Annotated[float, ParameterSpec()] = 0.0,
        g: Annotated[float, ParameterSpec()] = 9.81,
        length: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.rest = rest
        self.g = g
        self.length = length
