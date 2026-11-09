"""Define the Lennard-Jones pair potential."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import PairInteraction
from .....core.specs import ParameterSpec


class LennardJonesInteraction(PairInteraction):
    """Lennard-Jones pair potential without a shift.

    The potential is `4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)`.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the group or particle of the first members.

        second:
            The path of the group or particle of the second members.

        cutoff:
            The distance beyond which pairs do not interact.

        position:
            The name of the vector state that holds the positions of the members.

        write:
            The name of the state whose rate the force changes, the position of order 2 by default.

        divisor:
            The name of the parameter that holds the masses of the members, or `None` for unit masses.

        period:
            The side of a periodic box, for minimum image distances, or `None` for the period of the space.

        epsilon:
            The depth of the potential well.

        sigma:
            The distance at which the potential is zero.

    Examples:
        >>> from skcomplex.systems.physics.common.particles import LennardJonesInteraction
        >>> LennardJonesInteraction('lj', first='gas', second='gas', cutoff=2.5).sigma
        1.0
    """

    def __init__(
        self,
        label: str,
        first: str,
        second: str,
        cutoff: float = 1.0,
        position: str = 'x',
        *,
        write: str = 'x',
        divisor: str | None = None,
        period: float | None = None,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(
            label,
            first,
            second,
            cutoff=cutoff,
            position=position,
            write=write,
            potential=self._get_lennard_jones_energy,
            divisor=divisor,
            period=period,
        )
        self.epsilon = epsilon
        self.sigma = sigma

    @staticmethod
    def _get_lennard_jones_energy(s: SimpleNamespace, r: Expression) -> Expression:
        """Return `4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)`."""
        return 4 * s.epsilon * ((s.sigma / r) ** 12 - (s.sigma / r) ** 6)
