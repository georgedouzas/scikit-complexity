"""Motion of a parametric pendulum, the Euler-Lagrange equation of `θ'² / 2 + (g / l)(1 + ε cos Ω t) cos θ`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos

from ....core.components.interactions import EulerLagrangeInteraction
from ....core.components.laws.utils import derivative
from ....core.specs import RoleSpec


class ParametricPendulumInteraction(EulerLagrangeInteraction):
    """Motion of a parametric pendulum, the Euler-Lagrange equation of `θ'² / 2 + (g / l)(1 + ε cos Ω t) cos θ`.

    The Rayleigh dissipation `c θ'² / 2` adds the friction, so the angle follows
    `θ'' + c θ' + (g / l)(1 + ε cos Ω t) sin θ = 0`, with the parameters of the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the pendulum.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import ParametricPendulumInteraction
        >>> ParametricPendulumInteraction('swing', target='pendulum').states
        'theta'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, states='theta', dissipation=self._get_friction)

    @staticmethod
    def _get_friction(namespace: SimpleNamespace) -> Expression:
        """Return the Rayleigh dissipation `c θ'² / 2` of the friction."""
        return namespace.target.c * derivative(namespace.target.theta, 1) ** 2 / 2

    def _get_lagrangian(self, namespace: SimpleNamespace) -> Expression:
        """Return the Lagrangian of the parametric pendulum per moment of inertia.

        Args:
            namespace:
                The namespace of the symbols the Lagrangian can use.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        p = namespace.target
        gravity = p.g / p.l * (1 + p.epsilon * cos(p.Omega * namespace.t))
        return derivative(p.theta, 1) ** 2 / 2 + gravity * cos(p.theta)
