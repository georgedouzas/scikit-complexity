"""Macroeconomics systems: growth and business cycles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ....core.components.interactions import MapInteraction
from ....core.components.laws.utils import lag
from ....core.components.system import System
from ..._catalog import register_system
from ._economy import EconomyElement
from ._income import IncomeElement
from ._solow_growth import SolowGrowthInteraction


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
