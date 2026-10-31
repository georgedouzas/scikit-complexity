"""Define the spring between two particles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo, sqrt
from sage.sets.real_set import RealSet

from .....core.interactions import PotentialInteraction
from .....specs import ParameterSpec, RoleSpec


class SpringInteraction(PotentialInteraction):
    """Spring between two particles.

    The energy is `k * (|x_first - x_second| - length)**2 / 2`.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the first particle.

        second:
            The path of the second particle.

        k:
            The stiffness.

        length:
            The rest length.

        position:
            The name of the vector state that holds the positions of the particles.

        velocity:
            The name of the vector state that holds the velocities of the particles.

        mass:
            The name of the state or parameter that holds the masses of the particles.

    Examples:
        >>> from skcomplex.models.physics.common.particles import SpringInteraction
        >>> SpringInteraction('spring', first='a', second='b', k=3.0, length=1.0).length
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        length: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        position: str = 'x',
        velocity: str = 'v',
        mass: str = 'm',
    ) -> None:
        super().__init__(label, position=position, velocity=velocity, mass=mass)
        self.first = first
        self.second = second
        self.k = k
        self.length = length

    def _get_energy(self, s: SimpleNamespace) -> Expression:
        """Return `k * (|x_first - x_second| - length)**2 / 2`.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """
        gap = getattr(s.first, self.position) - getattr(s.second, self.position)
        return s.k * (sqrt(gap.dot_product(gap)) - s.length) ** 2 / 2
