"""Hops of one particle from every site of a ring, to the right at rate `p u(n)` and to the left at rate `q u(n)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.interactions import TransferInteraction
from ....specs import ParameterSpec, RoleSpec
from ._zero_range_hop import ZeroRangeHopInteraction


class ZeroRangeHopsInteraction(TransferInteraction):
    """Hops of one particle from every site of a ring, to the right at rate `p u(n)` and to the left at rate `q u(n)`.

    The rate depends only on the occupation `n` of the departure site, through `u(n) = 1 + b / n`. It is a transfer of
    one particle along every edge of the ring, to the right and back.

    Args:
        label:
            The label of the hops.

        sites:
            The group of sites, in ring order.

        p:
            The rate scale of hops to the right.

        q:
            The rate scale of hops to the left.

        b:
            The strength of the decrease of `u(n)` with the occupation, 0 for a constant rate.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import ZeroRangeHopsInteraction
        >>> ZeroRangeHopsInteraction('hops', sites='sites').b
        4.0
    """

    def __init__(
        self,
        label: str,
        *,
        sites: Annotated[str, RoleSpec()],
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        q: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
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
        )
        self.sites = sites
        self.p = p
        self.q = q
        self.b = b

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return the sites, whose every member hops to the next one and back."""
        return [(namespace.sites, namespace.sites)]

    @staticmethod
    def _get_right_rate(s: SimpleNamespace) -> Expression:
        """Return the rate `p u(n)` of a hop to the right."""
        return s.p * ZeroRangeHopInteraction._departure_rate(s.source.n, s.b)

    @staticmethod
    def _get_left_rate(s: SimpleNamespace) -> Expression:
        """Return the rate `q u(n)` of a hop to the left, from the site on the right."""
        return s.q * ZeroRangeHopInteraction._departure_rate(s.target.n, s.b)
