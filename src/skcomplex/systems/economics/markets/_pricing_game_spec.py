"""The discretised pricing game of a duopoly with loyal customers and switchers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass
from typing import cast

import numpy as np

from ....core.engine.solvers import SystemView
from ....core.specs import BaseSpec
from ....core.types import Array
from ._pricing_firm_view import PricingFirmView


@dataclass(frozen=True, eq=False)
class PricingGameSpec(BaseSpec):
    """Grid of prices of two firms, each with a loyal share, that share the switchers who pay the lower price.

    Args:
        prices:
            The prices the firms choose among, increasing.

        alpha:
            The loyal share of every firm.

        beta:
            The share of switchers.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.economics.markets import PricingGameSpec
        >>> PricingGameSpec(np.array([0.5, 1.0]), 0.25, 0.5).beta
        0.5
    """

    prices: Array
    alpha: float
    beta: float

    def utility(self, agent: PricingFirmView, system: SystemView, option: float) -> float:
        """Return what a price has earned so far against the prices of the rival, its score.

        Args:
            agent:
                The firm.

            system:
                The system view, which the score does not read.

            option:
                The price.

        Returns:
            utility:
                The score of the price.
        """
        return float(agent.scores[int(np.searchsorted(self.prices, option))])

    def payoff(self, agent: PricingFirmView, system: SystemView, k: int) -> float:
        """Return what the price of index `k` earns against the price the rival charged in this step.

        Args:
            agent:
                The firm.

            system:
                The system view, which holds the rival.

            k:
                The index of the price.

        Returns:
            payoff:
                The profit of the price.
        """
        rival = cast(PricingFirmView, next(item for item in system.agents(agent.group) if item.id != agent.id))
        price, against = float(self.prices[k]), float(rival.price)
        share = 1.0 if price < against else 0.5 if price == against else 0.0
        return price * (self.alpha + self.beta * share)
