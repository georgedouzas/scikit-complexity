"""Cart on a horizontal track at the position `y`, with a pendulum of angle `theta` hanging from it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo, pi
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


def _cart_and_bob(y: float, theta: float, l: float, **_: float) -> np.ndarray:
    """Return the positions of a cart on a horizontal track and of the bob that hangs from it."""
    return np.array([[y, 0.0], [y + l * np.sin(theta), -l * np.cos(theta)]])


class CartPendulumElement(Element):
    """Cart on a horizontal track at the position `y`, with a pendulum of angle `theta` hanging from it.

    The position and the angle have order 2. The motion comes from `CartPendulumInteraction`.

    Args:
        label:
            The label of the cart and pendulum.

        visual:
            How the cart and pendulum are drawn, or `None` for its class visual.

        y:
            The initial position of the cart.

        y_d1:
            The initial velocity of the cart.

        theta:
            The initial angle of the pendulum.

        theta_d1:
            The initial angular velocity of the pendulum.

        m1:
            The mass of the cart.

        m2:
            The mass of the bob.

        l:
            The length of the rod.

        g:
            The gravitational acceleration.

        F:
            A constant horizontal force on the cart.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import CartPendulumElement
        >>> CartPendulumElement('cart', m1=5.0).m1
        5.0
    """

    visual = VisualSpec('bodies', {'positions': _cart_and_bob, 'rods': True, 'traced': [1]})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y: Annotated[float, StateSpec(order=2)] = 0.0,
        y_d1: float = 0.0,
        theta: Annotated[float, StateSpec(period=2 * pi, order=2)] = 0.5,
        theta_d1: float = 0.0,
        m1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        m2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        l: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 9.81,
        F: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y = y
        self.y_d1 = y_d1
        self.theta = theta
        self.theta_d1 = theta_d1
        self.m1 = m1
        self.m2 = m2
        self.l = l
        self.g = g
        self.F = F
