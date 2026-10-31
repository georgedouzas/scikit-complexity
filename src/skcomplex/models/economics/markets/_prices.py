"""Markets models: prices that adjust continuously to excess demand."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import RateInteraction
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


def _excess_demand(s: SimpleNamespace) -> Expression:
    """Return the rate of the price, `k` times the demand `beta - lam P` less the supply `delta + alpha P + g P^2`."""
    market = s.target
    demand = market.beta - market.lam * market.P
    supply = market.delta + market.alpha * market.P + market.gamma * market.P**2
    return market.k * (demand - supply)


class QuadraticSupplyMarketElement(Element):
    """Market of one good whose price adjusts to the excess of a linear demand over a quadratic supply.

    Args:
        label:
            The label of the market.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        P:
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

    Examples:
        >>> from skcomplex.models.economics.markets import QuadraticSupplyMarketElement
        >>> QuadraticSupplyMarketElement('market', P=2.0).P
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        P: Annotated[float, StateSpec()] = 1.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        lam: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        delta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        gamma: Annotated[float, ParameterSpec()] = -0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.P = P
        self.k = k
        self.beta = beta
        self.lam = lam
        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma


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
    market = QuadraticSupplyMarketElement(
        'market',
        P=price,
        k=k,
        beta=beta,
        lam=lam,
        delta=delta,
        alpha=alpha,
        gamma=gamma,
    )
    adjustment = RateInteraction('adjustment', target='market', state='P', rate=_excess_demand)
    return System(elements=[market], interactions=[adjustment])
