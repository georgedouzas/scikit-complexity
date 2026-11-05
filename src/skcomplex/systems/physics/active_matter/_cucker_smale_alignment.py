"""Cucker-Smale alignment, `x_i'' += K / N psi(r_ij) (x_j' - x_i')` with `psi(r) = (1 + r^2)^(-beta)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import PairCouplingInteraction
from ....core.specs import ParameterSpec


class CuckerSmaleAlignmentInteraction(PairCouplingInteraction):
    """Cucker-Smale alignment, `x_i'' += K / N psi(r_ij) (x_j' - x_i')` with `psi(r) = (1 + r^2)^(-beta)`.

    Every pair of members within the cutoff pulls their velocities, the first derivatives of their positions of order
    2, towards each other with a weight that decays with their distance.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        first:
            The path of the group or particle of the first members.

        second:
            The path of the group or particle of the second members.

        cutoff:
            The distance beyond which pairs do not align, large for the all-to-all system.

        position:
            The name of the vector state of order 2 that holds the positions of the members.

        K:
            The strength of the alignment.

        beta:
            The decay exponent of the weight.

        period:
            The side of a periodic box, for minimum image distances, or `None` for the period of the space.

        n_agents:
            The number of agents `N` that normalizes the alignment.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import CuckerSmaleAlignmentInteraction
        >>> CuckerSmaleAlignmentInteraction('align', first='birds', second='birds', beta=0.25).beta
        0.25
    """

    def __init__(
        self,
        label: str,
        first: str,
        second: str,
        cutoff: float = 1.0,
        position: str = 'x',
        *,
        K: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.25,
        period: float | None = None,
        n_agents: int = 1,
    ) -> None:
        super().__init__(
            label,
            first,
            second,
            cutoff=cutoff,
            position=position,
            write=position,
            function=self._align,
            period=period,
        )
        self.K = K
        self.beta = beta
        self.n_agents = n_agents

    def _align(self, namespace: SimpleNamespace) -> Expression:
        """Return the pull `K / N psi(r) (x_j' - x_i')` of the velocity of the first member towards the second."""
        first, second = namespace.first, namespace.second
        separation = getattr(first, self.position) - getattr(second, self.position)
        velocity = f'{self.position}_d1'
        weight = namespace.K / self.n_agents * (1 + separation.dot_product(separation)) ** (-namespace.beta)
        return weight * (getattr(second, velocity) - getattr(first, velocity))
