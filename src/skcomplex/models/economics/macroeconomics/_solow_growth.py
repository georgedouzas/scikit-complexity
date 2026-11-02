"""Solow growth of the capital per worker `k` of a target, `dk/dt = s * k**alpha - delta * k`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.interactions import RateInteraction
from ....specs import ParameterSpec, RoleSpec


class SolowGrowthInteraction(RateInteraction):
    """Solow growth of the capital per worker `k` of a target, `dk/dt = s * k**alpha - delta * k`.

    A rate interaction on the state `k` whose rate is the saving of a Cobb-Douglas output minus depreciation.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the economy.

        s:
            The saving rate.

        alpha:
            The output elasticity of capital.

        delta:
            The depreciation rate.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import SolowGrowthInteraction
        >>> SolowGrowthInteraction('solow', target='country', s=0.3).s
        0.3
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        s: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
        delta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, target=target, state='k')
        self.s = s
        self.alpha = alpha
        self.delta = delta

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the rate `s * k**alpha - delta * k`.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespace of the target.

        Returns:
            rate:
                The rate of the capital per worker.
        """
        k = namespace.target.k
        rate: Expression = namespace.s * k**namespace.alpha - namespace.delta * k
        return rate
