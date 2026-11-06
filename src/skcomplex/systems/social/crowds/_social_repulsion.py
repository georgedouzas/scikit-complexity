"""Repulsion between two walkers, the force `strength * exp((2 * radius - r) / falloff)` of the social force system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import (
    PairCouplingInteraction,
)
from ....core.specs import ParameterSpec


class SocialRepulsionInteraction(PairCouplingInteraction):
    """Repulsion between two walkers, the force `strength * exp((2 * radius - r) / falloff)` of the social force system.

    It derives from the pair energy `strength * falloff * exp((2 * radius - r) / falloff)`.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the group or walker of the first members.

        second:
            The path of the group or walker of the second members.

        cutoff:
            The distance beyond which walkers do not interact.

        position:
            The name of the vector state that holds the positions of the walkers.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        strength:
            The strength of the repulsion.

        falloff:
            The distance over which the repulsion falls off.

        radius:
            The radius of a walker.

    Examples:
        >>> from skcomplex.systems.social.crowds import SocialRepulsionInteraction
        >>> SocialRepulsionInteraction('repulsion', first='walkers', second='walkers').strength
        2.0
    """

    def __init__(
        self,
        label: str,
        first: str,
        second: str,
        cutoff: float = 1.0,
        position: str = 'x',
        *,
        period: float | None = None,
        strength: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        falloff: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
        radius: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(
            label,
            first,
            second,
            cutoff=cutoff,
            position=position,
            potential=self._get_pair_energy,
            period=period,
        )
        self.strength = strength
        self.falloff = falloff
        self.radius = radius

    @staticmethod
    def _get_pair_energy(s: SimpleNamespace, r: Expression) -> Expression:
        """Return the pair energy `strength * falloff * exp((2 * radius - r) / falloff)`."""
        return s.strength * s.falloff * exp((2 * s.radius - r) / s.falloff)
