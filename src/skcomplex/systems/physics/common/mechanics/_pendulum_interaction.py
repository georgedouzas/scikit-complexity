"""Motion of a pendulum.

The motion is the Euler-Lagrange equation of `L = m l² θ'² / 2 + m g l cos θ`.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos

from .....core.components.interactions import EulerLagrangeInteraction
from .....core.components.laws.utils import derivative
from .....core.specs import RoleSpec


class PendulumInteraction(EulerLagrangeInteraction):
    """Motion of a pendulum.

    The motion is the Euler-Lagrange equation of `L = m l² θ'² / 2 + m g l cos θ`.

    The equation is Newton's law `m l² θ'' = -m g l sin θ` for the angle `θ` of the target, whose mass `m`, length
    `l`, and gravity `g` are parameters of the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the pendulum.

    Examples:
        >>> from skcomplex.systems.physics.common.mechanics import PendulumInteraction
        >>> PendulumInteraction('swing', target='bob').states
        'theta'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, states='theta')

    def _get_lagrangian(self, namespace: SimpleNamespace) -> Expression:
        """Return the Lagrangian `m l² θ'² / 2 + m g l cos θ` of the pendulum.

        Args:
            namespace:
                The namespace of the symbols the Lagrangian can use.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        bob = namespace.target
        return bob.m * bob.l**2 * derivative(bob.theta, 1) ** 2 / 2 + bob.m * bob.g * bob.l * cos(bob.theta)
