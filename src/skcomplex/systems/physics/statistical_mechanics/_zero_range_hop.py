"""Hop of one particle along a link, forward at rate `p u(n)` and backward at rate `q u(n')`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import TransferInteraction
from ....core.components.laws.utils import where
from ....core.specs import ParameterSpec, RoleSpec


class ZeroRangeHopInteraction(TransferInteraction):
    """Hop of one particle along a link, forward at rate `p u(n)` and backward at rate `q u(n')`.

    The rate depends only on the occupation of the departure site, through `u(n) = 1 + b / n`. It is a transfer of one
    particle from the first site to the second, with jumps back at the reverse rate.

    Args:
        label:
            The label of the link.

        link:
            The departure and the arrival site of a forward hop.

        p:
            The forward rate scale.

        q:
            The backward rate scale.

        b:
            The strength of the decrease of `u(n)` with the occupation, 0 for a constant rate.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ZeroRangeHopInteraction
        >>> ZeroRangeHopInteraction('hop', link=['sites[0]', 'sites[1]']).b
        4.0
    """

    def __init__(
        self,
        label: str,
        *,
        link: Annotated[list[str], RoleSpec(variadic=True)],
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        q: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
    ) -> None:
        super().__init__(
            label,
            source='',
            target='',
            state='n',
            amount=1.0,
            rate=self._get_forward_rate,
            reverse_rate=self._get_backward_rate,
        )
        self.link = link
        self.p = p
        self.q = q
        self.b = b

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return the two sites of the link, the departure and the arrival of a forward hop."""
        first, second = namespace.link
        return [(first, second)]

    @staticmethod
    def _get_forward_rate(s: SimpleNamespace) -> Expression:
        """Return the rate `p u(n)` of a hop from the first site."""
        return s.p * ZeroRangeHopInteraction._departure_rate(s.source.n, s.b)

    @staticmethod
    def _get_backward_rate(s: SimpleNamespace) -> Expression:
        """Return the rate `q u(n')` of a hop from the second site."""
        return s.q * ZeroRangeHopInteraction._departure_rate(s.target.n, s.b)

    @staticmethod
    def _departure_rate(n: Expression, b: Expression) -> Expression:
        """Return the zero-range rate `u(n) = 1 + b / n` of a site with `n` particles, or 0 for an empty site."""
        return where(n > 0, 1 + b / where(n > 0, n, 1), 0)
