"""Bilateral barter between paired traders, a transfer whose trades repeat while both traders gain."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import ClassVar

import numpy as np

from ....core.interactions import TransferInteraction
from ....core.laws import BaseLaw, RuleLaw
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ._exchange import trade_goods

GOODS = ('sugar', 'spice', 'salt')


class ExchangeInteraction(TransferInteraction):
    """Bilateral barter between paired traders, a transfer whose trades repeat while both traders gain.

    Every step the transfer pairs the traders, and every pair barters small quantities of every pair of goods at the
    geometric mean of their marginal rates of substitution while both gain. A trader holds every good in the state of
    its name and its Cobb-Douglas weight in the state `<good>_weight`. Every trade moves goods between the two traders,
    so the total of every good stays the same, which the default run checks as invariants. The exchange reports the
    mean `price` of the second good in units of the first, the standard deviation `dispersion` of its logarithm over
    the trades of the step, and the number of trades `volume`.

    Args:
        label:
            The label of the exchange.

        source:
            The path of the group of traders.

        target:
            The path of the group of traders, the source again.

        goods:
            The names of the traded goods.

        pairing:
            How the transfer pairs the traders, such as `'neighbor-pairs'` for disjoint pairs of neighbours or
            `'pairs'` for disjoint random pairs.

        quantity:
            The quantity of the bought good in every trade.

        phase:
            The phase the trades run in.

    Examples:
        >>> from skcomplex.models.economics.markets import ExchangeInteraction
        >>> ExchangeInteraction('exchange', source='traders', target='traders').reports
        ('price', 'dispersion', 'volume')
    """

    reports: ClassVar[tuple[str, ...]] = ('price', 'dispersion', 'volume')

    def __init__(
        self,
        label: str,
        *,
        source: str,
        target: str,
        goods: tuple[str, ...] = GOODS,
        pairing: str = 'pairs',
        quantity: float = 0.1,
        phase: str = 'trade',
    ) -> None:
        super().__init__(
            label,
            source=source,
            target=target,
            state=goods[0],
            amount=quantity,
            pairing=pairing,
            form='agent',
            phase=phase,
        )
        self.goods = goods
        self.quantity = quantity

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the barter of every pair of traders and the report of the trades of the step.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [*super()._get_laws(s), RuleLaw(self.report_trades, phase=self.phase)]

    def transfer(self, source: AgentViewProtocol, target: AgentViewProtocol, model: ModelViewProtocol) -> bool:
        """Barter between two traders while both gain, and keep the trades of the step for its report.

        Args:
            source:
                The first trader.

            target:
                The second trader.

            model:
                The model view of the exchange.

        Returns:
            traded:
                Whether the traders made a trade.
        """
        weights = [f'{good}_weight' for good in self.goods]
        trades = trade_goods(source, target, self.goods, weights, quantity=self.quantity)
        logs = model.storage.setdefault('trades', [])
        if isinstance(logs, list):
            logs.extend(trades)
        return bool(trades)

    def report_trades(self, model: ModelViewProtocol) -> None:
        """Report the mean price of the second good in the first, its dispersion, and the trades of the step.

        Args:
            model:
                The model view of the exchange.
        """
        trades = model.storage.pop('trades', [])
        trades = trades if isinstance(trades, list) else []
        logs = [math.log(price) for bought, paid, price in trades if (bought, paid) == (0, 1)]
        model.report('price', math.exp(float(np.mean(logs))) if logs else math.nan)
        model.report('dispersion', float(np.std(logs)) if len(logs) > 1 else math.nan)
        model.report('volume', len(trades))
