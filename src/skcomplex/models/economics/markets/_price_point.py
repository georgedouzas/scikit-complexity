"""A price of a mixed pricing equilibrium, with the share of rivals below it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.base import BasePart
from ....core.elements import Element
from ....core.laws import EquationLaw
from ....specs import ParameterSpec, StateSpec


class PricePointElement(Element):
    """Equilibrium distribution of prices at one price, which makes a firm indifferent among its prices.

    A firm that charges `P` sells to its loyal share `alpha`, and to the switchers `beta` when it undercuts all its
    `m - 1` rivals, each with the probability `1 - F(P)`, for the expected profit `P (alpha + beta (1 - F)^(m - 1))`.
    In equilibrium every price gives the profit `alpha` of selling to the loyal share at the highest price 1, which
    `F(P) = 1 - (alpha (1 - P) / (beta P))^(1 / (m - 1))` solves (Forrest 2018 Lemma 20.1).

    Args:
        label:
            The label of the point or the group.

        distribution:
            The initial guess of `F(P)`, which the equations replace.

        price:
            The price `P`.

        alpha:
            The loyal share of every firm.

        beta:
            The share of switchers.

        n_firms:
            The number of firms `m`.

    Examples:
        >>> from skcomplex.models.economics.markets import PricePointElement
        >>> PricePointElement('point', price=0.5).price
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        distribution: Annotated[float, StateSpec()] = 0.5,
        price: Annotated[float, StateSpec(domain=RealSet.open_closed(0, 1))] = 1.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.25,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        n_firms: Annotated[float, ParameterSpec(domain=RealSet.closed_open(2, oo))] = 2.0,
    ) -> None:
        super().__init__(label)
        self.distribution = distribution
        self.price = price
        self.alpha = alpha
        self.beta = beta
        self.n_firms = n_firms

    @staticmethod
    def get_profit(member: SimpleNamespace) -> Expression:
        """Return the expected profit of a firm at the price of a member, for the observables of a run.

        Args:
            member:
                The namespace of the states and parameters of the member.

        Returns:
            profit:
                The expected profit `P (alpha + beta (1 - F)^(m - 1))`.
        """
        return member.price * (member.alpha + member.beta * (1 - member.distribution) ** (member.n_firms - 1))

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the equation of the distribution.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equation of the distribution.
        """
        distribution = 1 - (s.alpha * (1 - s.price) / (s.beta * s.price)) ** (1 / (s.n_firms - 1))
        return [EquationLaw(s.distribution == distribution, target=s.distribution)]
