"""Food market in which people short of food, in random order, buy from the cheapest stocked shop within sight."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import ClassVar

from ....core.components.interactions import MatchingInteraction
from ....core.engine.solvers import SystemView
from ....core.specs import (
    MatchSpec,
)
from ._business_view import BusinessView


class FoodMarketInteraction(MatchingInteraction[SystemView]):
    """Food market in which people short of food, in random order, buy from the cheapest stocked shop within sight.

    A matching of the people, who request their `shortfall` of food for `reserve` steps within the budget of
    their money, with the firms, which provide their stock at their price. Every purchase moves the money to the firm,
    which books it as revenue, and the food to the buyer. Build it with
    `PartnerSpec(sight, 'score', score=FoodMarketInteraction.score, clearing='provider')`, whose score prefers the
    lowest price. Besides the reporters of a matching, the reporter `gdp` sums the value of
    the sales of a step.

    Examples:
        >>> from skcomplex.core.specs import PartnerSpec
        >>> from skcomplex.systems.economics.macroeconomics import FoodMarketInteraction
        >>> rule = PartnerSpec(5.0, 'score', score=FoodMarketInteraction.score, clearing='provider')
        >>> FoodMarketInteraction('shops', requesters='people', providers='businesses', rule=rule).reports[-1]
        'gdp'
    """

    reports: ClassVar[tuple[str, ...]] = (*MatchingInteraction.reports, 'gdp')

    @staticmethod
    def score(person: object, firm: object) -> float:
        """Return the score of a firm for a person, higher for a lower price.

        Args:
            person:
                The person that buys.

            firm:
                The firm that sells.

        Returns:
            score:
                Minus the price of the firm.
        """
        del person
        return -float(firm.price) if isinstance(firm, BusinessView) else -math.inf

    @staticmethod
    def sell(match: MatchSpec, system: SystemView) -> None:
        """Book the value of a sale as revenue of the firm.

        Args:
            match:
                The match of a buyer with a firm.

            system:
                The system view of the market.
        """
        del system
        if isinstance(match.provider, BusinessView):
            match.provider.revenue += match.quantity * match.value

    def _report_matches(self, matches: list[MatchSpec], unmatched: float, system: SystemView) -> None:
        """Report the price, the volume, the unmatched quantity, and the value of the sales of the step.

        Args:
            matches:
                The sales of the step.

            unmatched:
                The quantity of both sides left after the sales.

            system:
                The system view of the market.
        """
        super()._report_matches(matches, unmatched, system)
        system.report('gdp', sum(match.quantity * match.value for match in matches))
