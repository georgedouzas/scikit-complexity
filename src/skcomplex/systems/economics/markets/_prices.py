"""Markets systems: prices that adjust continuously to excess demand."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import RateInteraction
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class QuadraticSupplyMarketElement(Element):
    """Market of one good whose price adjusts to the excess demand.

    The excess is that of a linear demand over a quadratic supply.

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
        >>> from skcomplex.systems.economics.markets import QuadraticSupplyMarketElement
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


class RiccatiPricesSystem(BaseSystem):
    """A price that moves with excess demand.

    Demand is linear and supply is quadratic and saturating. The price follows the Riccati equation `dP/dt = k (D - S)`
    with demand `D = beta - lam P` and supply `S = delta + alpha P + gamma P^2`. When
    `(lam + alpha)^2 > 4 gamma (delta - beta)` there are two equilibrium prices, and prices below the larger one
    converge to the smaller one. Otherwise the price blows up in finite time. The system has the price `market.P`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.markets import RiccatiPricesSystem
        >>> RiccatiPricesSystem(gamma=-0.2).prepare().system_.elements[0].gamma
        -0.2
    """

    field = 'economics/markets'
    topics = ('market', 'riccati', 'ode')
    references = ('Forrest 2018',)
    until = 10.0
    dt = 0.1

    def __init__(
        self,
        price: float = 1.0,
        *,
        k: float = 1.0,
        beta: float = 10.0,
        lam: float = 1.0,
        delta: float = 1.0,
        alpha: float = 2.0,
        gamma: float = -0.1,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.price = price
        self.k = k
        self.beta = beta
        self.lam = lam
        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def _excess_demand(s: SimpleNamespace) -> Expression:
        """Return the rate of the price.

        It is `k` times the demand `beta - lam P` less supply `delta + alpha P + g P^2`.
        """
        market = s.target
        demand = market.beta - market.lam * market.P
        supply = market.delta + market.alpha * market.P + market.gamma * market.P**2
        return market.k * (demand - supply)

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        market = QuadraticSupplyMarketElement(
            'market',
            P=self.price,
            k=self.k,
            beta=self.beta,
            lam=self.lam,
            delta=self.delta,
            alpha=self.alpha,
            gamma=self.gamma,
        )
        adjustment = RateInteraction('adjustment', target='market', state='P', rate=self._excess_demand)
        return System(self.label, elements=[market], interactions=[adjustment])
