"""Macroeconomics systems: growth and business cycles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import MapInteraction, RateInteraction
from ....core.components.laws.utils import lag
from ....core.components.system import System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system


class EconomyElement(Element):
    """Economy described by its capital per worker.

    Args:
        label:
            The label of the economy.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        k:
            The initial capital per worker.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import EconomyElement
        >>> EconomyElement('country', k=1.0).k
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        k: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.k = k


class IncomeElement(Element):
    """Economy described by its national income, which keeps the income of the previous period.

    Args:
        label:
            The label of the economy.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        Y:
            The income of the current period.

        Y_lag1:
            The income of the previous period.

        c:
            The marginal propensity to consume.

        v:
            The accelerator, the investment response to the change in consumption.

        G:
            The government spending.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import IncomeElement
        >>> IncomeElement('economy', c=0.5).c
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        Y: Annotated[object, StateSpec(lags=1, initial=(100.0,))] = 110.0,
        Y_lag1: float = 100.0,
        c: Annotated[object, ParameterSpec()] = 0.8,
        v: Annotated[object, ParameterSpec()] = 1.5,
        G: Annotated[object, ParameterSpec()] = 20.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.Y = Y
        self.Y_lag1 = Y_lag1
        self.c = c
        self.v = v
        self.G = G


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
        >>> from skcomplex.systems.economics.macroeconomics import SolowGrowthInteraction
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


@register_system(
    field='economics/macroeconomics',
    tags=('growth', 'ode'),
    references=('Solow 1956',),
    defaults={'until': 200.0, 'dt': 1.0},
)
def solow(k: float = 1.0, *, s: float = 0.2, alpha: float = 0.3, delta: float = 0.05) -> System:
    """Return an economy whose capital per worker approaches its steady state through saving and depreciation.

    Args:
        k:
            The initial capital per worker.

        s:
            The saving rate.

        alpha:
            The capital share of output.

        delta:
            The depreciation rate.

    Returns:
        system:
            The system, with the economy `economy`.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import solow
        >>> solow(s=0.3).interactions[0].s
        0.3
    """
    return System(
        elements=[EconomyElement('economy', k=k)],
        interactions=[SolowGrowthInteraction('growth', target='economy', s=s, alpha=alpha, delta=delta)],
    )


def _next_income(s: SimpleNamespace) -> Expression:
    """Return the income of the next period, `c (1 + v) Y_t - c v Y_{t-1} + G`."""
    economy = s.target
    income: Expression = economy.c * (1 + economy.v) * economy.Y - economy.c * economy.v * lag(economy.Y, 1) + economy.G
    return income


@register_system(
    field='economics/macroeconomics',
    tags=('business cycle', 'discrete map', 'second order'),
    references=('Samuelson 1939',),
    defaults={'until': 40, 'dt': 1},
)
def multiplier_accelerator(
    income: float = 110.0,
    previous: float = 100.0,
    *,
    c: float = 0.8,
    v: float = 1.5,
    G: float = 20.0,
) -> System:
    """Return Samuelson's multiplier-accelerator, `Y_{t+1} = c (1 + v) Y_t - c v Y_{t-1} + G`.

    Income oscillates around `G / (1 - c)`, with growing amplitude when `c v > 1` and complex characteristic roots.

    Args:
        income:
            The income of the current period.

        previous:
            The income of the previous period.

        c:
            The marginal propensity to consume.

        v:
            The accelerator, the investment response to the change in consumption.

        G:
            The government spending.

    Returns:
        system:
            The system, with the income `economy.Y` and its previous value `economy.Y_lag1`.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import multiplier_accelerator
        >>> multiplier_accelerator(v=0.5).elements[0].v
        0.5
    """
    return System(
        elements=[IncomeElement('economy', Y=income, Y_lag1=previous, c=c, v=v, G=G)],
        interactions=[MapInteraction('cycle', target='economy', state='Y', function=_next_income)],
    )
