"""Define the swing of a pendulum.

It is the Euler-Lagrange equation of a bob on a rigid rod under gravity.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, cos

from .....core.specs import Role
from ._lagrangian import LagrangianInteraction


class PendulumSwingInteraction(LagrangianInteraction):
    """Motion of a pendulum, the Euler-Lagrange equation of `L = m l² θ'² / 2 + m g l cos θ`.

    The equation is Newton's law `m l² θ'' = -m g l sin θ` for the angle `θ` of the target, whose mass `m`, length
    `l`, and gravity `g` are values of the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the pendulum.

    Examples:
        >>> from skcomplex.systems.physics.common.mechanics import PendulumSwingInteraction
        >>> PendulumSwingInteraction('swing', target='bob').target
        'bob'
    """

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _get_coordinates(self, context: SimpleNamespace) -> list[Expression]:
        """Return the angle of the pendulum."""
        return [context.target.theta]

    def _get_lagrangian(
        self,
        context: SimpleNamespace,
        positions: list[Expression],
        velocities: list[Expression],
    ) -> Expression:
        """Return the Lagrangian `m l² θ'² / 2 + m g l cos θ` of the pendulum.

        Args:
            context:
                The namespace of the pendulum and the time `t`.

            positions:
                The symbol of the angle.

            velocities:
                The symbol of the angular velocity.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        bob = context.target
        return bob.m * bob.l**2 * velocities[0] ** 2 / 2 + bob.m * bob.g * bob.l * cos(positions[0])
