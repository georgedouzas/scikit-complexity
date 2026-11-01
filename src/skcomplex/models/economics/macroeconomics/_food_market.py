"""Food market in which people short of food, in random order, buy from the cheapest stocked shop within sight."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ....core.interactions import MatchingInteraction
from ....engine.solvers import ModelView
from ....specs import (
    MatchSpec,
)
from ._business_view import BusinessView


class FoodMarketInteraction(MatchingInteraction[ModelView]):
    """Food market in which people short of food, in random order, buy from the cheapest stocked shop within sight.

    A greedy matching of the people, who request their `shortfall` of food for `reserve` steps within the budget of
    their money, with the firms, which provide their stock at their price. Every purchase moves the money to the firm,
    which books it as revenue, and the food to the buyer. Build it with `GreedyMatchingSpec(candidates=sight)`, whose
    default score prefers the lowest price. Besides the reporters of a matching, the reporter `gdp` sums the value of
    the sales of a step.

    Examples:
        >>> from skcomplex.specs import GreedyMatchingSpec
        >>> from skcomplex.models.economics.macroeconomics import FoodMarketInteraction
        >>> FoodMarketInteraction(
        ...     'shops', requesters='people', providers='businesses', rule=GreedyMatchingSpec(candidates=5.0)
        ... ).reports[-1]
        'gdp'
    """

    reports: ClassVar[tuple[str, ...]] = (*MatchingInteraction.reports, 'gdp')

    @staticmethod
    def sell(match: MatchSpec, model: ModelView) -> None:
        """Book the value of a sale as revenue of the firm.

        Args:
            match:
                The match of a buyer with a firm.

            model:
                The model view of the market.
        """
        del model
        if isinstance(match.provider, BusinessView):
            match.provider.revenue += match.quantity * match.price

    def _report_matches(self, matches: list[MatchSpec], unmatched: float, model: ModelView) -> None:
        """Report the price, the volume, the unmatched quantity, and the value of the sales of the step.

        Args:
            matches:
                The sales of the step.

            unmatched:
                The quantity of both sides left after the sales.

            model:
                The model view of the market.
        """
        super()._report_matches(matches, unmatched, model)
        model.report('gdp', sum(match.quantity * match.price for match in matches))
