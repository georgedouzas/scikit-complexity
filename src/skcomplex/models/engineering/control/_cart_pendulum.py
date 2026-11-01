"""Inverted pendulum on a cart that a bounded horizontal force pushes, after Åström and Murray."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo, sin
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, EquationLaw
from ....core.laws.utils import derivative
from ....specs import ParameterSpec, StateSpec, VisualSpec

GRAVITY = 9.81


FORCE_LIMIT = 50.0


class CartPendulumElement(Element):
    """Inverted pendulum on a cart that a bounded horizontal force pushes, after Åström and Murray.

    The cart of mass `M` at position `x` carries a point mass `m` at the end of a rod of length `l`, at the angle
    `theta` from upright, positive towards negative `x`. The equations of motion are
    `(M + m) x'' - m l theta'' cos(theta) + m l theta'^2 sin(theta) = force` and
    `l theta'' - x'' cos(theta) = g sin(theta)`, so upright is unstable and a positive force tilts the pendulum back
    from a positive angle.

    Args:
        label:
            The label of the cart.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position of the cart.

        x_d1:
            The initial velocity of the cart.

        theta:
            The initial angle from upright.

        theta_d1:
            The initial angular velocity.

        force:
            The horizontal force on the cart, the control input.

        M:
            The mass of the cart.

        m:
            The mass at the end of the rod.

        l:
            The length of the rod.

        g:
            The acceleration of gravity.

    Examples:
        >>> from skcomplex.models.engineering.control import CartPendulumElement
        >>> CartPendulumElement('cart', theta=0.1).theta
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        theta: Annotated[float, StateSpec(order=2)] = 0.0,
        theta_d1: float = 0.0,
        force: Annotated[float, ParameterSpec(adaptive=True, bounds=(-FORCE_LIMIT, FORCE_LIMIT))] = 0.0,
        M: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        l: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = GRAVITY,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.theta = theta
        self.theta_d1 = theta_d1
        self.force = force
        self.M = M
        self.m = m
        self.l = l
        self.g = g

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the equations of motion of the cart and the pendulum.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        x_acceleration, theta_acceleration = derivative(s.x, 2), derivative(s.theta, 2)
        spin = derivative(s.theta, 1)
        return [
            EquationLaw(
                (s.M + s.m) * x_acceleration
                - s.m * s.l * theta_acceleration * cos(s.theta)
                + s.m * s.l * spin**2 * sin(s.theta)
                == s.force,
            ),
            EquationLaw(s.l * theta_acceleration - x_acceleration * cos(s.theta) == s.g * sin(s.theta)),
        ]
