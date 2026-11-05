"""Hops of a particle from every site of a line to an empty neighbour, to the right at rate `p` and left at `q`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import TransferInteraction
from ....core.specs import ParameterSpec, RoleSpec


class ExclusionHopsInteraction(TransferInteraction):
    """Hops of a particle from every site of a line to an empty neighbour, to the right at rate `p` and left at `q`.

    No particle hops past the ends of the line. It is a transfer of one particle along every edge of the line, to the
    right and back, from an occupied site to an empty one.

    Args:
        label:
            The label of the hops.

        sites:
            The group of sites, in line order.

        p:
            The rate of hops to the right.

        q:
            The rate of hops to the left.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ExclusionHopsInteraction
        >>> ExclusionHopsInteraction('hops', sites='lane').p
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        sites: Annotated[str, RoleSpec()],
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        q: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(
            label,
            source=sites,
            target=sites,
            state='n',
            amount=1.0,
            rate=self._get_right_rate,
            reverse_rate=self._get_left_rate,
            offset=1,
            boundary='fixed',
        )
        self.sites = sites
        self.p = p
        self.q = q

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return the sites, whose every member hops to the next one and back."""
        return [(namespace.sites, namespace.sites)]

    @staticmethod
    def _get_right_rate(s: SimpleNamespace) -> Expression:
        """Return the rate of a hop to the right, `p` from an occupied site to an empty one."""
        return s.p * s.source.n * (1 - s.target.n)

    @staticmethod
    def _get_left_rate(s: SimpleNamespace) -> Expression:
        """Return the rate of a hop to the left, `q` from an occupied site to an empty one."""
        return s.q * s.target.n * (1 - s.source.n)
