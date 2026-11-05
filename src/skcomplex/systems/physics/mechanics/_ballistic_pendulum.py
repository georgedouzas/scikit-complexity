"""Bullet of mass `m` at the position `x` flying towards the block of mass `M` of a pendulum with the angle `theta`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo, pi
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class BallisticPendulumElement(Element):
    """Bullet of mass `m` at the position `x` flying towards the block of mass `M` of a pendulum with the angle `theta`.

    The position and the angle have order 2. The bullet flies freely, the pendulum swings, and the impact transfers
    the momentum of the bullet to the pendulum.

    Args:
        label:
            The label of the bullet and pendulum.

        visual:
            How the bullet and pendulum are drawn, or `None` for none.

        x:
            The initial position of the bullet, negative before the block.

        x_d1:
            The initial velocity of the bullet.

        theta:
            The initial angle of the pendulum.

        theta_d1:
            The initial angular velocity of the pendulum.

        m:
            The mass of the bullet.

        M:
            The mass of the block.

        l:
            The length of the rod.

        g:
            The gravitational acceleration.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import BallisticPendulumElement
        >>> BallisticPendulumElement('pendulum', M=2.0).M
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = -1.0,
        x_d1: float = 0.0,
        theta: Annotated[float, StateSpec(period=2 * pi, order=2)] = 0.0,
        theta_d1: float = 0.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        M: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        l: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 9.81,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.theta = theta
        self.theta_d1 = theta_d1
        self.m = m
        self.M = M
        self.l = l
        self.g = g
