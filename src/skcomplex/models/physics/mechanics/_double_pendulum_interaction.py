"""Motion of a double pendulum, the Euler-Lagrange equations of its kinetic energy and gravity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos

from ....core.interactions import EulerLagrangeInteraction
from ....core.laws.utils import derivative
from ....specs import RoleSpec


class DoublePendulumInteraction(EulerLagrangeInteraction):
    """Motion of a double pendulum, the Euler-Lagrange equations of its kinetic energy and gravity.

    The Lagrangian is `(m1 + m2) l1² θ1'² / 2 + m2 l2² θ2'² / 2 + m2 l1 l2 θ1' θ2' cos(θ1 - θ2) + (m1 + m2) g l1 cos θ1
    + m2 g l2 cos θ2`, with the masses, lengths, and gravity of the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the double pendulum.

    Examples:
        >>> from skcomplex.models.physics.mechanics import DoublePendulumInteraction
        >>> DoublePendulumInteraction('motion', target='pendulum').states
        ('theta1', 'theta2')
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, states=('theta1', 'theta2'))

    def _get_lagrangian(self, namespace: SimpleNamespace) -> Expression:
        """Return the Lagrangian of the double pendulum.

        Args:
            namespace:
                The namespace of the symbols the Lagrangian can use.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        p = namespace.target
        w1, w2 = derivative(p.theta1, 1), derivative(p.theta2, 1)
        kinetic = (p.m1 + p.m2) * p.l1**2 * w1**2 / 2 + p.m2 * p.l2**2 * w2**2 / 2
        kinetic += p.m2 * p.l1 * p.l2 * w1 * w2 * cos(p.theta1 - p.theta2)
        return kinetic + (p.m1 + p.m2) * p.g * p.l1 * cos(p.theta1) + p.m2 * p.g * p.l2 * cos(p.theta2)
