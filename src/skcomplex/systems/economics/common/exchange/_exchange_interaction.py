"""Bilateral barter between paired traders.

It is a transfer whose trades repeat while both traders gain.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import ClassVar

import numpy as np

from .....core.components.base import BaseComponent
from .....core.components.interactions import Interaction
from .....core.components.laws import PartnerSpec, UpdateLaw
from .....core.components.laws.utils import pairs
from .....core.components.spaces import BaseSpace
from .....core.components.types import AgentViewProtocol, SystemViewProtocol
from .....core.specs import Role
from ._exchange import trade_goods

GOODS = ('sugar', 'spice', 'salt')
PAIRINGS = {
    'pairs': PartnerSpec('all', 'uniform', capacity=1),
    'neighbor-pairs': PartnerSpec('neighbors', 'uniform', capacity=1),
}


class ExchangeInteraction(Interaction):
    """Bilateral barter between paired traders.

    Its trades repeat while both traders gain. Every step the exchange pairs the traders, and every
    pair barters small quantities of every pair of goods at the geometric mean of their marginal rates of substitution
    while both gain. A trader holds every good in the state of its name and its Cobb-Douglas weight in the state
    `<good>_weight`. Every trade moves goods between the two traders, so the total of every good stays the same, which
    the default run checks as invariants. The exchange reports the mean `price` of the second good in units of the
    first, the standard deviation `dispersion` of its logarithm over the trades of the step, and the number of trades
    `volume`.

    Args:
        label:
            The label of the exchange.

        target:
            The path of the group of traders.

        goods:
            The names of the traded goods.

        pairing:
            How the exchange pairs the traders, `'neighbor-pairs'` for disjoint pairs of neighbours or `'pairs'` for
            disjoint random pairs.

        quantity:
            The quantity of the bought good in every trade.

        phase:
            The phase the trades run in.

    Examples:
        >>> from skcomplex.systems.economics.markets import ExchangeInteraction
        >>> ExchangeInteraction('exchange', target='traders').reports
        ('price', 'dispersion', 'volume')
    """

    law_classes = (UpdateLaw,)

    reports: ClassVar[tuple[str, ...]] = ('price', 'dispersion', 'volume')

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        goods: tuple[str, ...] = GOODS,
        pairing: str = 'pairs',
        quantity: float = 0.1,
        phase: str = 'trade',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.goods = goods
        self.pairing = pairing
        self.quantity = quantity
        self.phase = phase

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the barter of every pair of traders and then the report of the trades as rules of the system."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [
                UpdateLaw(context, self.trade, phase=self.phase),
                UpdateLaw(context, self.report_trades, phase=self.phase),
            ]

    def trade(self, system: SystemViewProtocol) -> None:
        """Pair the traders and let every pair barter, in random order.

        Args:
            system:
                The system view of the exchange.
        """
        traders = system.get_agents('target')
        found = list(pairs(traders, traders, system.rng, PAIRINGS[self.pairing]))
        for index in system.rng.permutation(len(found)):
            first, second = found[int(index)]
            if first.alive and second.alive:
                self.transfer(first, second, system)

    def transfer(self, source: AgentViewProtocol, target: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Barter between two traders while both gain.

        It keeps the trades of the step for its report.

        Args:
            source:
                The first trader.

            target:
                The second trader.

            system:
                The system view of the exchange.

        Returns:
            traded:
                Whether the traders made a trade.
        """
        weights = [f'{good}_weight' for good in self.goods]
        trades = trade_goods(source, target, self.goods, weights, quantity=self.quantity)
        logs = system.storage.setdefault('trades', [])
        if isinstance(logs, list):
            logs.extend(trades)
        return bool(trades)

    def report_trades(self, system: SystemViewProtocol) -> None:
        """Report the mean price of the second good in the first, and the trades.

        It also reports the dispersion of the price, and the trades are those of the step.

        Args:
            system:
                The system view of the exchange.
        """
        trades = system.storage.pop('trades', [])
        trades = trades if isinstance(trades, list) else []
        logs = [math.log(price) for bought, paid, price in trades if (bought, paid) == (0, 1)]
        system.report('price', math.exp(float(np.mean(logs))) if logs else math.nan)
        system.report('dispersion', float(np.std(logs)) if len(logs) > 1 else math.nan)
        system.report('volume', len(trades))
