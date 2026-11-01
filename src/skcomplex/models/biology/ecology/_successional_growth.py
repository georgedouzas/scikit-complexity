"""Logistic growth of intermediate or climax trees, crowded by their own group and another, with seedling input."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.interactions import RateInteraction
from ....specs import ParameterSpec, RoleSpec
from ._pioneer_growth import PioneerGrowthInteraction


class SuccessionalGrowthInteraction(RateInteraction):
    """Logistic growth of intermediate or climax trees, crowded by their own group and another, with seedling input.

    The rate is `dV/dt = a + g V (1 - w_1 V_1 / V_mat,1 - w_2 V_2 / V_mat,2)`, with weights normalized to sum to one.
    One of the two crowding stands is usually the target itself.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the growing stand.

        first:
            The path of the first crowding stand.

        second:
            The path of the second crowding stand.

        g:
            The specific growth rate.

        w_first:
            The weight of the first crowding stand.

        w_second:
            The weight of the second crowding stand.

        seedlings:
            The constant input `a` of seedlings that averts extinction.

    Examples:
        >>> from skcomplex.models.biology.ecology import SuccessionalGrowthInteraction
        >>> SuccessionalGrowthInteraction('growth', target='gewa', first='gewa', second='sundari', g=0.1).g
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        g: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        w_first: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        w_second: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        seedlings: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label, target=target, state='volume')
        self.first = first
        self.second = second
        self.g = g
        self.w_first = w_first
        self.w_second = w_second
        self.seedlings = seedlings

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the crowded logistic growth with the seedling input.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the stands.

        Returns:
            rate:
                The rate of the volume of the target.
        """
        s = namespace
        crowding = PioneerGrowthInteraction._crowding(s.first, s.second, s.w_first, s.w_second)
        rate: Expression = s.seedlings + s.g * s.target.volume * (1 - crowding)
        return rate
