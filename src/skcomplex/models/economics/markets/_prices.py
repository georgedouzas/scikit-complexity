"""Markets models: prices that adjust continuously to excess demand."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.runtime import register_model


class QuadraticSupplyMarketElement(Element):
    """Quadratic supply market element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        P: Annotated[object, StateSpec()] = 1.0,
        k: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        beta: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        lam: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        delta: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        alpha: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        gamma: Annotated[object, ParameterSpec()] = -0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.P = P
        self.k = k
        self.beta = beta
        self.lam = lam
        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the quadratic supply market element."""
        return [RateLaw(s.P, s.k * ((s.beta - s.lam * s.P) - (s.delta + s.alpha * s.P + s.gamma * s.P**2)))]


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
    return System(elements=[market])
