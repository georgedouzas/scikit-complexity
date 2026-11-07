"""Motion of a pendulum whose suspension point oscillates vertically or horizontally, an Euler-Lagrange equation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos, sin

from ....core.components.interactions import EulerLagrangeInteraction
from ....core.components.laws.utils import derivative
from ....core.errors import ValidationError
from ....core.specs import RoleSpec

DIRECTIONS = ('vertical', 'horizontal')


class ParametricPendulumInteraction(EulerLagrangeInteraction):
    """Motion of a pendulum whose suspension point oscillates vertically or horizontally, an Euler-Lagrange equation.

    The suspension point accelerates by `ε g cos Ω t`. Vertically, the Lagrangian per moment of inertia is
    `θ'² / 2 + (g / l)(1 + ε cos Ω t) cos θ`, and the acceleration modulates gravity, a parametric forcing.
    Horizontally, it is `θ'² / 2 + (g / l)(cos θ - ε cos Ω t sin θ)`, and the acceleration pushes the bob sideways, an
    external forcing. The Rayleigh dissipation `c θ'² / 2` adds the friction, so the angle follows
    `θ'' + c θ' + (g / l)(1 + ε cos Ω t) sin θ = 0` or `θ'' + c θ' + (g / l)(sin θ + ε cos Ω t cos θ) = 0`, with the
    parameters of the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the pendulum.

        direction:
            The direction in which the suspension point oscillates, `'vertical'` or `'horizontal'`.

    Raises:
        ValidationError:
            The direction is unknown.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import ParametricPendulumInteraction
        >>> ParametricPendulumInteraction('swing', target='pendulum').states
        'theta'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], direction: str = 'vertical') -> None:
        super().__init__(label, target=target, states='theta', dissipation=self._get_friction)
        self.direction = direction

    def _check(self) -> None:
        """Raise an error for an unknown direction of the suspension point.

        Raises:
            ValidationError:
                The direction is unknown.
        """
        super()._check()
        if self.direction not in DIRECTIONS:
            error_msg = f'Unknown direction {self.direction!r}. Use one of {", ".join(DIRECTIONS)}.'
            raise ValidationError(error_msg)

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
        drive = p.epsilon * cos(p.Omega * namespace.t)
        if self.direction == 'horizontal':
            return derivative(p.theta, 1) ** 2 / 2 + p.g / p.l * (cos(p.theta) - drive * sin(p.theta))
        return derivative(p.theta, 1) ** 2 / 2 + p.g / p.l * (1 + drive) * cos(p.theta)
