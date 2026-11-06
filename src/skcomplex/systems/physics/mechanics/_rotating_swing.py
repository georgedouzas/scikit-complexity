"""Rotating swing: chairs on chains hanging from the rods of a stand that turns about a vertical axis."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo, pi
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class RotatingSwingElement(Element):
    """Rotating swing: chairs on chains hanging from the rods of a stand that turns about a vertical axis.

    The angle `phi` of the stand and the angle `theta` of the chains from the vertical have order 2. The motion comes
    from `RotatingSwingInteraction`.

    Args:
        label:
            The label of the swing.

        visual:
            How the swing is drawn, or `None` for none.

        phi:
            The initial angle of the stand.

        phi_d1:
            The initial angular velocity of the stand.

        theta:
            The initial angle of the chains.

        theta_d1:
            The initial angular velocity of the chains.

        J:
            The moment of inertia of the stand.

        m:
            The mass of every chair.

        n_chairs:
            The number of chairs.

        a:
            The length of the rods.

        l:
            The length of the chains.

        g:
            The gravitational acceleration.

        torque:
            The torque that drives the stand.

        friction:
            The friction coefficient of the stand's axis.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import RotatingSwingElement
        >>> RotatingSwingElement('swing', n_chairs=4.0).n_chairs
        4.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[float, StateSpec(period=2 * pi, order=2)] = 0.0,
        phi_d1: float = 0.0,
        theta: Annotated[float, StateSpec(order=2)] = 0.1,
        theta_d1: float = 0.0,
        J: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        n_chairs: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 8.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        l: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 9.81,
        torque: Annotated[float, ParameterSpec()] = 0.0,
        friction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi
        self.phi_d1 = phi_d1
        self.theta = theta
        self.theta_d1 = theta_d1
        self.J = J
        self.m = m
        self.n_chairs = n_chairs
        self.a = a
        self.l = l
        self.g = g
        self.torque = torque
        self.friction = friction
