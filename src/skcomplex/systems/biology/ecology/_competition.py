"""Lotka-Volterra competition of two species for the same resources."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, RoleSpec


class CompetitionInteraction(Interaction):
    """Lotka-Volterra competition of two species for the same resources.

    Each species grows logistically, and the other species fills its carrying capacity in proportion to a competition
    coefficient, `dn1/dt = r1 n1 (1 - (n1 + alpha12 n2) / K1)` and `dn2/dt = r2 n2 (1 - (n2 + alpha21 n1) / K2)`.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the first population.

        second:
            The path of the second population.

        r1:
            The growth rate of the first population.

        r2:
            The growth rate of the second population.

        K1:
            The carrying capacity of the first population.

        K2:
            The carrying capacity of the second population.

        alpha12:
            The effect of one member of the second population on the first, relative to one of its own.

        alpha21:
            The effect of one member of the first population on the second, relative to one of its own.

    Examples:
        >>> from skcomplex.systems.biology.ecology import CompetitionInteraction
        >>> CompetitionInteraction('competition', first='deer', second='elk', alpha12=0.5).alpha12
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        K2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        alpha12: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        alpha21: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label)
        self.first = first
        self.second = second
        self.r1 = r1
        self.r2 = r2
        self.K1 = K1
        self.K2 = K2
        self.alpha12 = alpha12
        self.alpha21 = alpha21

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the logistic rates with the competitor in the crowding term.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        first, second = s.first.n, s.second.n
        return [
            RateLaw(first, s.r1 * first * (1 - (first + s.alpha12 * second) / s.K1)),
            RateLaw(second, s.r2 * second * (1 - (second + s.alpha21 * first) / s.K2)),
        ]
