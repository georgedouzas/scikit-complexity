"""Markets models: price dynamics of supply and demand."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import MapInteraction
from ....core.system import System
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model


@register_model(
    field='economics/markets',
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
        >>> from skcomplex.models.economics.markets import cobweb
        >>> cobweb(price=2.0).elements[0].price
        2.0
    """
    return System(
        elements=[MarketElement('market', price=price)],
        interactions=[CobwebInteraction('cobweb', target='market', a=a, b=b, c=c, d=d)],
    )


class MarketElement(Element):
    """Market of one good described by its price.

    Args:
        label:
            The label of the market.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        price:
            The initial price.

    Examples:
        >>> from skcomplex.models.economics.markets import MarketElement
        >>> MarketElement('wheat', price=2.0).price
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        price: Annotated[float, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.price = price


class CobwebInteraction(MapInteraction):
    """Cobweb model, where supply reacts to the previous price and demand clears the market.

    Demand is `a - b * price` and supply is `c + d * previous price`, so the next price is
    `(a - c - d * price) / b`, a map of the state `price` of the market.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the market.

        a:
            The demand intercept.

        b:
            The demand slope.

        c:
            The supply intercept.

        d:
            The supply slope.

    Examples:
        >>> from skcomplex.models.economics.markets import CobwebInteraction
        >>> CobwebInteraction('cobweb', target='wheat', a=10, b=2, c=1, d=1).b
        2
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        a: Annotated[float, ParameterSpec()] = 10.0,
        b: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        c: Annotated[float, ParameterSpec()] = 1.0,
        d: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, target=target, state='price', function=self._get_price)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    @staticmethod
    def _get_price(s: SimpleNamespace) -> Expression:
        """Return the next price `(a - c - d * price) / b`, at which demand buys the supply of the last price."""
        return (s.a - s.c - s.d * s.target.price) / s.b
