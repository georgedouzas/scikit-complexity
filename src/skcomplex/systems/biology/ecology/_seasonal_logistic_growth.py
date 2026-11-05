"""Logistic growth under a carrying capacity that varies periodically, `K(t) = K (1 + amplitude sin(2 pi t / T))`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.specs import ParameterSpec, RoleSpec
from ..common.populations import LogisticGrowthInteraction


class SeasonalLogisticGrowthInteraction(LogisticGrowthInteraction):
    """Logistic growth under a carrying capacity that varies periodically, `K(t) = K (1 + amplitude sin(2 pi t / T))`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the population.

        r:
            The per capita growth rate at low density.

        K:
            The mean carrying capacity.

        amplitude:
            The relative amplitude of the variation of the capacity, below 1.

        period:
            The period `T` of the variation.

    Examples:
        >>> from skcomplex.systems.biology.ecology import SeasonalLogisticGrowthInteraction
        >>> SeasonalLogisticGrowthInteraction('growth', target='rabbits', period=12.0).period
        12.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        amplitude: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        period: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, target=target, r=r, K=K)
        self.amplitude = amplitude
        self.period = period

    def _get_capacity(self, namespace: SimpleNamespace) -> Expression:
        """Return the capacity `K (1 + amplitude sin(2 pi t / T))` at the time of the namespace.

        Args:
            namespace:
                The namespace of the symbols the rate can use.

        Returns:
            capacity:
                The carrying capacity at the time `t`.
        """
        s = namespace
        capacity: Expression = s.K * (1 + s.amplitude * sin(2 * pi * s.t / s.period))
        return capacity
