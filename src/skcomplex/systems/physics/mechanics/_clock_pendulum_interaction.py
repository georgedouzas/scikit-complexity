"""Motion of a pendulum clock on a moving support, the Euler-Lagrange equations with the escapement."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos, sign

from ....core.components.interactions import EulerLagrangeInteraction
from ....core.components.laws.utils import derivative
from ....core.specs import RoleSpec


class ClockPendulumInteraction(EulerLagrangeInteraction):
    """Motion of a pendulum clock on a moving support, the Euler-Lagrange equations with the escapement.

    The Lagrangian is `(M + m) x'² / 2 + m l x' θ' cos θ + m l² θ'² / 2 + m g l cos θ`. The Rayleigh dissipation
    `λ l sgn(|θ| - θ_c) θ'² + rho x'²` damps the support and gives the escapement `2 λ l sgn(|θ| - θ_c) θ'`, which pumps
    energy into the pendulum below the angle `θ_c` and takes it out beyond.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the clock.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import ClockPendulumInteraction
        >>> ClockPendulumInteraction('motion', target='clock').states
        ('theta', 'x')
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, states=('theta', 'x'), dissipation=self._get_escapement)

    @staticmethod
    def _get_escapement(namespace: SimpleNamespace) -> Expression:
        """Return the Rayleigh dissipation of the escapement and of the damping of the support."""
        p = namespace.target
        escapement = p.lam * p.l * sign(abs(p.theta) - p.theta_c) * derivative(p.theta, 1) ** 2
        return escapement + p.rho * derivative(p.x, 1) ** 2

    def _get_lagrangian(self, namespace: SimpleNamespace) -> Expression:
        """Return the Lagrangian of the pendulum and its support.

        Args:
            namespace:
                The namespace of the symbols the Lagrangian can use.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        p = namespace.target
        v, w = derivative(p.x, 1), derivative(p.theta, 1)
        kinetic = (p.M + p.m) * v**2 / 2 + p.m * p.l * v * w * cos(p.theta) + p.m * p.l**2 * w**2 / 2
        return kinetic + p.m * p.g * p.l * cos(p.theta)
