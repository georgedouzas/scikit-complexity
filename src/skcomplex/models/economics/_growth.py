"""Model economic growth and market price dynamics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...core import Domain, parameter, role, state
from ...model import Element, Interaction, Law, Rate, Symbols, Update


class Economy(Element):
    """Economy described by its capital per worker.

    Args:
        label:
            The label of the economy.

        k:
            The initial capital per worker.

    Examples:
        >>> from skcomplex.models.economics import Economy
        >>> Economy('country', k=1.0).k
        1.0
    """

    k: float = state(default=1.0, domain=Domain.NONNEGATIVE)


class SolowGrowth(Interaction):
    """Solow growth, `dk/dt = s * k**alpha - delta * k`.

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
        >>> from skcomplex.models.economics import SolowGrowth
        >>> SolowGrowth('solow', target='country', s=0.3).s
        0.3
    """

    target: str = role()
    s: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    alpha: float = parameter(default=0.3, domain=Domain.POSITIVE)
    delta: float = parameter(default=0.05, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate `s * k**alpha - delta * k`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.target.k, s.s * s.target.k**s.alpha - s.delta * s.target.k)]


class Market(Element):
    """Market of one good described by its price.

    Args:
        label:
            The label of the market.

        price:
            The initial price.

    Examples:
        >>> from skcomplex.models.economics import Market
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
        >>> from skcomplex.models.economics import Cobweb
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
