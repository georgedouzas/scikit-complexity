"""Markets models: price dynamics of supply and demand."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, role, state
from ....core.model import Element, Interaction, Law, Symbols, System, Update
from ....core.runtime import register_model


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
        elements=[Market('market', price=price)],
        interactions=[Cobweb('cobweb', target='market', a=a, b=b, c=c, d=d)],
    )


class Market(Element):
    """Market of one good described by its price.

    Args:
        label:
            The label of the market.

        price:
            The initial price.

    Examples:
        >>> from skcomplex.models.economics.markets import Market
        >>> Market('wheat', price=2.0).price
        2.0
    """

    price: float = state(default=1.0, domain=Domain.REAL)


class Cobweb(Interaction):
    """Cobweb model, where supply reacts to the previous price and demand clears the market.

    Demand is `a - b * price` and supply is `c + d * previous price`, so the next price is
    `(a - c - d * price) / b`.

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
        >>> from skcomplex.models.economics.markets import Cobweb
        >>> Cobweb('cobweb', target='wheat', a=10, b=2, c=1, d=1).b
        2
    """

    target: str = role()
    a: float = parameter(default=10.0, domain=Domain.REAL)
    b: float = parameter(default=2.0, domain=Domain.POSITIVE)
    c: float = parameter(default=1.0, domain=Domain.REAL)
    d: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the next price `(a - c - d * price) / b`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Update(s.target.price, (s.a - s.c - s.d * s.target.price) / s.b)]
