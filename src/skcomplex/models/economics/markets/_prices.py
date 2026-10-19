"""Markets models: prices that adjust continuously to excess demand."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, state
from ....core.model import Element, Rate, System
from ....core.runtime import register_model

QuadraticSupplyMarket = Element.define(
    'QuadraticSupplyMarket',
    states={'P': state(default=1.0, domain=Domain.REAL)},
    parameters={
        'k': parameter(default=1.0, domain=Domain.POSITIVE),
        'beta': parameter(default=10.0, domain=Domain.POSITIVE),
        'lam': parameter(default=1.0, domain=Domain.POSITIVE),
        'delta': parameter(default=1.0, domain=Domain.POSITIVE),
        'alpha': parameter(default=2.0, domain=Domain.POSITIVE),
        'gamma': parameter(default=-0.1, domain=Domain.REAL),
    },
    laws=lambda s: [Rate(s.P, s.k * ((s.beta - s.lam * s.P) - (s.delta + s.alpha * s.P + s.gamma * s.P**2)))],
)


@register_model(
    field='economics/markets',
    tags=('market', 'riccati', 'ode'),
    references=('Forrest 2018',),
    defaults={'until': 10.0, 'dt': 0.1},
)
def riccati_prices(
    price: float = 1.0,
    *,
    k: float = 1.0,
    beta: float = 10.0,
    lam: float = 1.0,
    delta: float = 1.0,
    alpha: float = 2.0,
    gamma: float = -0.1,
) -> System:
    """Return a price that moves with excess demand under linear demand and saturating quadratic supply.

    The price follows the Riccati equation `dP/dt = k (D - S)` with demand `D = beta - lam P` and supply
    `S = delta + alpha P + gamma P^2`. When `(lam + alpha)^2 > 4 gamma (delta - beta)` there are two equilibrium prices,
    and prices below the larger one converge to the smaller one. Otherwise the price blows up in finite time.

    Args:
        price:
            The initial price.

        k:
            The speed of the price adjustment.

        beta:
            The demand at price zero.

        lam:
            The slope of the demand.

        delta:
            The supply at price zero.

        alpha:
            The linear intensity of the supply.

        gamma:
            The quadratic intensity of the supply, negative for saturation.

    Returns:
        system:
            The model, with the price `market.P`.

    Examples:
        >>> from skcomplex.models.economics.markets import riccati_prices
        >>> riccati_prices(gamma=-0.2).elements[0].gamma
        -0.2
    """
    market = QuadraticSupplyMarket('market', P=price, k=k, beta=beta, lam=lam, delta=delta, alpha=alpha, gamma=gamma)
    return System(elements=[market])
