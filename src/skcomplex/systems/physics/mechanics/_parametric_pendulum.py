"""Pendulum whose suspension point oscillates vertically, with the angle `theta` of order 2."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

from sage.all__sagemath_symbolics import oo, pi
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ParametricPendulumElement(Element):
    """Pendulum whose suspension point oscillates vertically, with the angle `theta` of order 2.

    The motion comes from `ParametricPendulumInteraction`.

    Args:
        label:
            The label of the pendulum.

        visual:
            How the pendulum is drawn, or `None` for none.

        theta:
            The initial angle.

        theta_d1:
            The initial angular velocity.

        g:
            The gravitational acceleration.

        l:
            The length of the rod.

        epsilon:
            The amplitude of the acceleration of the suspension point, relative to `g`.

        Omega:
            The angular frequency of the suspension point.

        c:
            The friction coefficient.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import ParametricPendulumElement
        >>> ParametricPendulumElement('swing', epsilon=0.5).epsilon
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi, order=2)] = 0.1,
        theta_d1: float = 0.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 9.81,
        l: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        epsilon: Annotated[float, ParameterSpec()] = 0.2,
        Omega: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2 * math.sqrt(9.81),
        c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.g = g
        self.l = l
        self.epsilon = epsilon
        self.Omega = Omega
        self.c = c
