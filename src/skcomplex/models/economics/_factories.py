"""Build the ready-made economics models."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...core import state
from ...model import Element, System, Update, lag
from ...runtime import register_model
from ._growth import Cobweb, Economy, Market, SolowGrowth


@register_model(
    field='economics',
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
            The model, with the economy `economy`.

    Examples:
        >>> from skcomplex.models.economics import solow
        >>> solow(s=0.3).interactions[0].s
        0.3
    """
    return System(
        elements=[Economy('economy', k=k)],
        interactions=[SolowGrowth('growth', target='economy', s=s, alpha=alpha, delta=delta)],
    )


@register_model(
    field='economics',
    tags=('market', 'discrete map'),
    references=('Ezekiel 1938',),
    defaults={'until': 30, 'dt': 1},
)
def cobweb(price: float = 1.0, *, a: float = 10.0, b: float = 2.0, c: float = 1.0, d: float = 1.0) -> System:
    """Return a market whose price converges to or oscillates around equilibrium as supply lags the price.

    Args:
        price:
            The initial price.

        a:
            The demand intercept.

        b:
            The demand slope.

        c:
            The supply intercept.

        d:
            The supply slope.

    Returns:
        system:
            The model, with the market `market`.

    Examples:
        >>> from skcomplex.models.economics import cobweb
        >>> cobweb(price=2.0).elements[0].price
        2.0
    """
    return System(
        elements=[Market('market', price=price)],
        interactions=[Cobweb('cobweb', target='market', a=a, b=b, c=c, d=d)],
    )


Income = Element.define(
    'Income',
    states={'Y': state(default=110.0, lags=1, initial=(100.0,))},
    parameters={'c': 0.8, 'v': 1.5, 'G': 20.0},
    laws=lambda s: [Update(s.Y, s.c * (1 + s.v) * s.Y - s.c * s.v * lag(s.Y, 1) + s.G)],
)


@register_model(
    field='economics',
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
            The model, with the income `economy.Y` and its previous value `economy.Y_lag1`.

    Examples:
        >>> from skcomplex.models.economics import multiplier_accelerator
        >>> multiplier_accelerator(v=0.5).elements[0].v
        0.5
    """
    return System(elements=[Income('economy', Y=income, Y_lag1=previous, c=c, v=v, G=G)])
