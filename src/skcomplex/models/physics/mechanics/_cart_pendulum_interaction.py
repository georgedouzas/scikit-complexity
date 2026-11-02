"""Motion of a cart and the pendulum that hangs from it, the Euler-Lagrange equations with a force on the cart."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos

from ....core.interactions import EulerLagrangeInteraction
from ....core.laws.utils import derivative
from ....specs import RoleSpec


class CartPendulumInteraction(EulerLagrangeInteraction):
    """Motion of a cart and the pendulum that hangs from it, the Euler-Lagrange equations with a force on the cart.

    The Lagrangian is `(m1 + m2) y'² / 2 + m2 l y' θ' cos θ + m2 l² θ'² / 2 + m2 g l cos θ`, and the force `F` of the
    target drives the position `y` of the cart.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the cart and pendulum.

    Examples:
        >>> from skcomplex.models.physics.mechanics import CartPendulumInteraction
        >>> CartPendulumInteraction('motion', target='cart').states
        ('y', 'theta')
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, states=('y', 'theta'), forcing=self._get_forces_on_coordinates)

    @staticmethod
    def _get_forces_on_coordinates(namespace: SimpleNamespace) -> list[Expression]:
        """Return the generalized forces, the force `F` on the cart and none on the angle."""
        return [namespace.target.F, 0]

    def _get_lagrangian(self, namespace: SimpleNamespace) -> Expression:
        """Return the Lagrangian of the cart and pendulum.

        Args:
            namespace:
                The namespace of the symbols the Lagrangian can use.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        p = namespace.target
        v, w = derivative(p.y, 1), derivative(p.theta, 1)
        kinetic = (p.m1 + p.m2) * v**2 / 2 + p.m2 * p.l * v * w * cos(p.theta) + p.m2 * p.l**2 * w**2 / 2
        return kinetic + p.m2 * p.g * p.l * cos(p.theta)
