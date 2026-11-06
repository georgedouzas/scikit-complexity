"""Ball of mass `m` on a rod at the distance `r` from the joint, turning about a vertical axis by the angle `theta`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

from sage.all__sagemath_symbolics import oo, pi
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class RotatingBallElement(Element):
    """Ball of mass `m` on a rod at the distance `r` from the joint, turning about a vertical axis by the angle `theta`.

    The rod is fixed at the angle `beta` to the axis, so the ball is at the distance `r sin β` from it, and `H` is its
    angular momentum about the axis.

    Args:
        label:
            The label of the ball.

        visual:
            How the ball is drawn, or `None` for none.

        theta:
            The initial angle about the axis.

        m:
            The mass of the ball.

        r:
            The distance of the ball from the joint of the rod.

        beta:
            The angle of the rod to the axis, in radians.

        H:
            The angular momentum of the ball about the axis.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import RotatingBallElement
        >>> RotatingBallElement('ball', H=2.0).H
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        r: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = math.pi / 2,
        H: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.m = m
        self.r = r
        self.beta = beta
        self.H = H
