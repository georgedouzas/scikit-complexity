"""Define Newtonian gravity between two particles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo, sqrt
from sage.sets.real_set import RealSet

from .....specs import ParameterSpec, RoleSpec
from ._potential_energy import PotentialEnergyInteraction


class GravityInteraction(PotentialEnergyInteraction):
    """Newtonian gravity between two particles.

    The energy is `-G * m_first * m_second / |x_first - x_second|`, and its gradient moves both particles.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the first particle.

        second:
            The path of the second particle.

        G:
            The gravitational constant.

        states:
            The name of the vector state that holds the positions of the particles.

        divisor:
            The name of the state or parameter that holds the masses of the particles.

    Examples:
        >>> from skcomplex.models.physics.common.particles import GravityInteraction
        >>> GravityInteraction('gravity', first='sun', second='earth', G=1.0).G
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        G: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        states: str = 'x',
        divisor: str = 'm',
    ) -> None:
        super().__init__(label, target=first, states=states, divisor=divisor)
        self.first = first
        self.second = second
        self.G = G

    def _get_potential(self, s: SimpleNamespace) -> Expression:
        """Return `-G * m_first * m_second / |x_first - x_second|`.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """
        gap = getattr(s.first, self._get_names()[0]) - getattr(s.second, self._get_names()[0])
        masses = getattr(s.first, str(self.divisor)) * getattr(s.second, str(self.divisor))
        return -s.G * masses / sqrt(gap.dot_product(gap))
