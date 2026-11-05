"""Motion of a rotating swing, the Euler-Lagrange equations with a torque and friction on the stand."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos, sin

from ....core.components.interactions import EulerLagrangeInteraction
from ....core.components.laws.utils import derivative
from ....core.specs import RoleSpec


class RotatingSwingInteraction(EulerLagrangeInteraction):
    """Motion of a rotating swing, the Euler-Lagrange equations with a torque and friction on the stand.

    The Lagrangian is `[J + n m (a + l sin θ)²] φ'² / 2 + n m l² θ'² / 2 + n m g l cos θ`, the torque drives the angle
    `φ` of the stand, and the Rayleigh dissipation `c φ'² / 2` adds the friction of its axis.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the swing.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import RotatingSwingInteraction
        >>> RotatingSwingInteraction('motion', target='swing').states
        ('phi', 'theta')
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(
            label,
            target=target,
            states=('phi', 'theta'),
            dissipation=self._get_friction,
            forcing=self._get_torques,
        )

    @staticmethod
    def _get_friction(namespace: SimpleNamespace) -> Expression:
        """Return the Rayleigh dissipation `c φ'² / 2` of the stand's axis."""
        return namespace.target.friction * derivative(namespace.target.phi, 1) ** 2 / 2

    @staticmethod
    def _get_torques(namespace: SimpleNamespace) -> list[Expression]:
        """Return the generalized forces, the torque on the stand and none on the chains."""
        return [namespace.target.torque, 0]

    def _get_lagrangian(self, namespace: SimpleNamespace) -> Expression:
        """Return the Lagrangian of the stand and the chairs.

        Args:
            namespace:
                The namespace of the symbols the Lagrangian can use.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        p = namespace.target
        chairs = p.n_chairs * p.m
        radius = p.a + p.l * sin(p.theta)
        kinetic = (p.J + chairs * radius**2) * derivative(p.phi, 1) ** 2 / 2
        kinetic += chairs * p.l**2 * derivative(p.theta, 1) ** 2 / 2
        return kinetic + chairs * p.g * p.l * cos(p.theta)
