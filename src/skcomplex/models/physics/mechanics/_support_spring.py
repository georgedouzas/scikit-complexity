"""Spring of stiffness `k` between the supports of two pendulum clocks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.interactions import EulerLagrangeInteraction
from ....specs import ParameterSpec, RoleSpec


class SupportSpringInteraction(EulerLagrangeInteraction):
    """Spring of stiffness `k` between the supports of two pendulum clocks.

    The spring adds the Lagrangian `-k (x2 - x1)² / 2` to the Euler-Lagrange equations of the supports, so every clock
    answers with the accelerations of its support and its pendulum through the mass matrix of its own Lagrangian.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the first clock.

        second:
            The path of the second clock.

        k:
            The stiffness of the spring.

    Examples:
        >>> from skcomplex.models.physics.mechanics import SupportSpringInteraction
        >>> SupportSpringInteraction('spring', first='clock_0', second='clock_1', k=2.0).k
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
        super().__init__(label, target=first, states='x')
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
        return -namespace.k * (namespace.second.x - namespace.first.x) ** 2 / 2
