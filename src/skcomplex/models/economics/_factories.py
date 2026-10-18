"""Build the ready-made economics models."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...model import System
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
