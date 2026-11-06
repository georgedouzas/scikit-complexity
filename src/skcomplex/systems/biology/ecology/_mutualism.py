"""Mutualism with a handling time, where encounters feed both species, `a b n1 n2 / (1 + a T n)` each."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, RoleSpec


class MutualismInteraction(Interaction):
    """Mutualism with a handling time, where encounters feed both species, `a b n1 n2 / (1 + a T n)` each.

    The benefit to each species saturates with the abundance of its partner, as the handling time `T` limits how much
    of the partner's resource it can use. A handling time of 0 gives the unbounded type I mutualism `a b n1 n2`.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the first population.

        second:
            The path of the second population.

        a:
            The effective search rate.

        b:
            The rate of encounters.

        handling:
            The handling time `T` of the resource of the partner.

    Examples:
        >>> from skcomplex.systems.biology.ecology import MutualismInteraction
        >>> MutualismInteraction('pollination', first='orchids', second='bees', handling=0.5).handling
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        handling: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label)
        self.first = first
        self.second = second
        self.a = a
        self.b = b
        self.handling = handling

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the saturating benefits `a b n1 n2 / (1 + a T n2)` and `a b n1 n2 / (1 + a T n1)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        first, second = s.first.n, s.second.n
        encounters = s.a * s.b * first * second
        return [
            RateLaw(first, encounters / (1 + s.a * s.handling * second)),
            RateLaw(second, encounters / (1 + s.a * s.handling * first)),
        ]
