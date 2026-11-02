"""Attraction at long range and repulsion at short range, `C_r exp(-r / l_r) - C_a exp(-r / l_a)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.interactions import PairCouplingInteraction
from ....specs import ParameterSpec


class MorsePotentialInteraction(PairCouplingInteraction):
    """Attraction at long range and repulsion at short range, `C_r exp(-r / l_r) - C_a exp(-r / l_a)`.

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
            The name of the position state of the members.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        C_a:
            The strength of the attraction.

        l_a:
            The range of the attraction.

        C_r:
            The strength of the repulsion.

        l_r:
            The range of the repulsion.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import MorsePotentialInteraction
        >>> MorsePotentialInteraction('flocking', first='birds', second='birds', cutoff=10.0).l_a
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
        C_a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        l_a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        C_r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        l_r: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
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
        self.C_a = C_a
        self.l_a = l_a
        self.C_r = C_r
        self.l_r = l_r

    @staticmethod
    def _get_pair_energy(s: SimpleNamespace, r: Expression) -> Expression:
        """Return `C_r exp(-r / l_r) - C_a exp(-r / l_a)`."""
        return s.C_r * exp(-r / s.l_r) - s.C_a * exp(-r / s.l_a)
