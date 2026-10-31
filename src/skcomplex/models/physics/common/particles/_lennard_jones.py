"""Define the Lennard-Jones pair potential."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from .....core.interactions import PairForceInteraction
from .....specs import ParameterSpec


class LennardJonesInteraction(PairForceInteraction):
    """Lennard-Jones pair potential, `4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)`, without a shift.

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

        velocity:
            The name of the vector state that holds the velocities of the members.

        mass:
            The name of the state or parameter that holds the masses of the members.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        epsilon:
            The depth of the potential well.

        sigma:
            The distance at which the potential is zero.

    Examples:
        >>> from skcomplex.models.physics.common.particles import LennardJonesInteraction
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
        velocity: str = 'v',
        mass: str = 'm',
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
            velocity=velocity,
            mass=mass,
            period=period,
            energy=self._get_lennard_jones_energy,
        )
        self.epsilon = epsilon
        self.sigma = sigma

    @staticmethod
    def _get_lennard_jones_energy(s: SimpleNamespace, r: Expression) -> Expression:
        """Return `4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)`."""
        return 4 * s.epsilon * ((s.sigma / r) ** 12 - (s.sigma / r) ** 6)
