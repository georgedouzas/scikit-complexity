"""Pendulum clock, a pendulum of angle `theta` hanging from a support at the horizontal position `x`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ClockPendulumElement(Element):
    """Pendulum clock, a pendulum of angle `theta` hanging from a support at the horizontal position `x`.

    The angle and the position have order 2. The motion comes from `ClockPendulumInteraction`.

    Args:
        label:
            The label of the clock.

        visual:
            How the clock is drawn, or `None` for none.

        theta:
            The initial angle of the pendulum.

        theta_d1:
            The initial angular velocity of the pendulum.

        x:
            The initial position of the support.

        x_d1:
            The initial velocity of the support.

        m:
            The mass of the bob.

        l:
            The length of the pendulum.

        M:
            The mass of the support.

        g:
            The gravitational acceleration.

        rho:
            The damping of the support.

        lam:
            The strength of the escapement.

        theta_c:
            The angle below which the escapement pumps energy in.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import ClockPendulumElement
        >>> ClockPendulumElement('clock', lam=0.2).lam
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[float, StateSpec(order=2)] = 0.2,
        theta_d1: float = 0.0,
        x: Annotated[float, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        l: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        M: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 9.8,
        rho: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        lam: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        theta_c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.x = x
        self.x_d1 = x_d1
        self.m = m
        self.l = l
        self.M = M
        self.g = g
        self.rho = rho
        self.lam = lam
        self.theta_c = theta_c
