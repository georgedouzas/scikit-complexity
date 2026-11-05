"""Phoretic attraction between active colloids, the pair potential `-v0 R^2 / r` of an overdamped position."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import PairCouplingInteraction
from ....core.specs import ParameterSpec


class PhoreticAttractionInteraction(PairCouplingInteraction):
    """Phoretic attraction between active colloids, the pair potential `-v0 R^2 / r` of an overdamped position.

    Every member of a pair drifts towards the other at the speed `v0 R^2 / r^2`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        first:
            The path of the group or particle of the first members.

        second:
            The path of the group or particle of the second members.

        cutoff:
            The distance beyond which the attraction is neglected.

        position:
            The name of the vector state that holds the positions of the members.

        v0:
            The drift speed at the distance `R`.

        R:
            The particle radius.

        period:
            The side of a periodic box, for minimum image distances, or `None` for the period of the space.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import PhoreticAttractionInteraction
        >>> PhoreticAttractionInteraction('phoresis', first='colloids', second='colloids', v0=20.0).v0
        20.0
    """

    def __init__(
        self,
        label: str,
        first: str,
        second: str,
        cutoff: float = 1.0,
        position: str = 'x',
        *,
        v0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        R: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        period: float | None = None,
    ) -> None:
        super().__init__(
            label,
            first,
            second,
            cutoff=cutoff,
            position=position,
            write=position,
            potential=self._attraction,
            period=period,
        )
        self.v0 = v0
        self.R = R

    @staticmethod
    def _attraction(namespace: SimpleNamespace, r: Expression) -> Expression:
        """Return the potential `-v0 R^2 / r` of a pair at the distance `r`."""
        return -namespace.v0 * namespace.R**2 / r
