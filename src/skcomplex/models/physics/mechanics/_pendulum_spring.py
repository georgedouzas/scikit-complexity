"""Spring between the bobs of two pendulums, which stretches with the horizontal distance of the bobs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo, sin
from sage.sets.real_set import RealSet

from ....core.interactions import EulerLagrangeInteraction
from ....specs import ParameterSpec, RoleSpec


class PendulumSpringInteraction(EulerLagrangeInteraction):
    """Spring between the bobs of two pendulums, which stretches with the horizontal distance of the bobs.

    The spring of stiffness `k` adds the Lagrangian `-k (l2 sin θ2 - l1 sin θ1)² / 2` to the Euler-Lagrange equations
    of the two angles, so its torque on every pendulum divides by the moment of inertia `m l²` of the pendulum's own
    Lagrangian.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the first pendulum, with the angle `theta`, the mass `m`, and the length `l`.

        second:
            The path of the second pendulum.

        k:
            The stiffness of the spring.

    Examples:
        >>> from skcomplex.models.physics.mechanics import PendulumSpringInteraction
        >>> PendulumSpringInteraction('spring', first='left', second='right', k=2.0).k
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, target=first, states='theta')
        self.first = first
        self.second = second
        self.k = k

    def _get_lagrangian(self, namespace: SimpleNamespace) -> Expression:
        """Return minus the energy of the spring.

        Args:
            namespace:
                The namespace of the symbols the Lagrangian can use, with `first` and `second`.

        Returns:
            lagrangian:
                The Lagrangian of the spring.
        """
        first, second = namespace.first, namespace.second
        stretch = second.l * sin(second.theta) - first.l * sin(first.theta)
        return -namespace.k * stretch**2 / 2
