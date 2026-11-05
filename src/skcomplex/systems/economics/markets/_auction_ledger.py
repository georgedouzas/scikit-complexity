"""Ledger that closes every trading period: it measures its surplus and gives every trader a new unit."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np

from ....core.components.interactions import FeedbackInteraction
from ....core.engine.solvers import SystemView
from ._zero_intelligence_buyer_view import ZeroIntelligenceBuyerView
from ._zero_intelligence_seller_view import ZeroIntelligenceSellerView


def maximum_surplus(values: np.ndarray, costs: np.ndarray) -> float:
    """Return the largest total surplus of single units, the competitive surplus of the supply and demand schedules.

    Args:
        values:
            The value of a unit to every buyer.

        costs:
            The cost of a unit to every seller.

    Returns:
        surplus:
            The sum of the positive differences between the highest values and the lowest costs.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.economics.markets import maximum_surplus
        >>> maximum_surplus(np.array([10.0, 6.0, 2.0]), np.array([1.0, 4.0, 8.0]))
        11.0
    """
    demand, supply = np.sort(values)[::-1], np.sort(costs)
    size = min(len(demand), len(supply))
    return float(np.maximum(demand[:size] - supply[:size], 0).sum())


class AuctionLedgerInteraction(FeedbackInteraction[SystemView]):
    """Ledger that closes every trading period: it measures its surplus and gives every trader a new unit.

    The realised surplus is the value of the units bought minus the cost of the units sold, and the allocative
    efficiency is its ratio to the maximum surplus of the supply and demand schedules.

    Args:
        label:
            The label of the ledger.

        phase:
            The phase the ledger acts in.

    Examples:
        >>> from skcomplex.systems.economics.markets import AuctionLedgerInteraction
        >>> AuctionLedgerInteraction('ledger', phase='close').reports
        ('surplus', 'efficiency')
    """

    reports: ClassVar[tuple[str, ...]] = ('surplus', 'efficiency')

    def act(self, system: SystemView) -> None:
        """Report the surplus and the efficiency of the trading period and open the next one.

        Args:
            system:
                The system view scoped to the ledger.
        """
        buyers = system.agents('buyers', ZeroIntelligenceBuyerView)
        sellers = system.agents('sellers', ZeroIntelligenceSellerView)
        values = np.array([buyer.value for buyer in buyers])
        costs = np.array([seller.value for seller in sellers])
        bought = np.array([buyer.demand <= 0 for buyer in buyers])
        sold = np.array([seller.supply <= 0 for seller in sellers])
        surplus = float(values[bought].sum() - costs[sold].sum())
        system.report('surplus', surplus)
        system.report('efficiency', surplus / maximum_surplus(values, costs))
        for buyer in buyers:
            buyer.demand = 1.0
        for seller in sellers:
            seller.supply = 1.0
