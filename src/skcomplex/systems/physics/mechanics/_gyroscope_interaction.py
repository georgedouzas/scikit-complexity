"""Motion of a gimballed gyroscope, the Euler-Lagrange equations of its kinetic energy with a torque on `beta`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos, sin

from ....core.components.interactions import EulerLagrangeInteraction
from ....core.components.laws.utils import derivative
from ....core.specs import RoleSpec


class GyroscopeInteraction(EulerLagrangeInteraction):
    """Motion of a gimballed gyroscope, the Euler-Lagrange equations of its kinetic energy with a torque on `beta`.

    The Lagrangian is the kinetic energy `[A2 + (A + A1) cos² β + (C + C1) sin² β] alpha'² / 2 + (A + B1) β'² / 2
    + C alpha' gamma' sin β + C gamma'² / 2`, and the torque of the target acts on the axis of the inner ring.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the gyroscope.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import GyroscopeInteraction
        >>> GyroscopeInteraction('motion', target='gyroscope').states
        ('alpha', 'beta', 'gamma')
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, states=('alpha', 'beta', 'gamma'), forcing=self._get_torques)

    @staticmethod
    def _get_torques(namespace: SimpleNamespace) -> list[Expression]:
        """Return the generalized forces, the torque on the inner ring and none on the outer ring and the rotor."""
        return [0, namespace.target.torque, 0]

    def _get_lagrangian(self, namespace: SimpleNamespace) -> Expression:
        """Return the kinetic energy of the rings and the rotor.

        Args:
            namespace:
                The namespace of the symbols the Lagrangian can use.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        p = namespace.target
        da, db, dc = derivative(p.alpha, 1), derivative(p.beta, 1), derivative(p.gamma, 1)
        outer = p.A2 + (p.A + p.A1) * cos(p.beta) ** 2 + (p.C + p.C1) * sin(p.beta) ** 2
        return outer * da**2 / 2 + (p.A + p.B1) * db**2 / 2 + p.C * da * dc * sin(p.beta) + p.C * dc**2 / 2
