"""Markets models: double auctions of zero-intelligence traders with and without budget constraints."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import MapInteraction, MatchingInteraction
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import (
    BatchMatchingSpec,
    ContinuousMatchingSpec,
    MatchSpec,
    PhaseSpec,
    PhasesScheduleSpec,
)
from ..._catalog import register_model
from ._auction_ledger import AuctionLedgerInteraction
from ._zero_intelligence_buyer import ZeroIntelligenceBuyerElement
from ._zero_intelligence_buyer_view import ZeroIntelligenceBuyerView
from ._zero_intelligence_seller import ZeroIntelligenceSellerElement
from ._zero_intelligence_seller_view import ZeroIntelligenceSellerView

PROTOCOLS = ('order_book', 'call_auction')


def _book_profit(trade: MatchSpec, model: ModelView) -> None:
    """Add the surplus of a trade to the profits of the buyer and the seller."""
    del model
    buyer, seller = trade.requester, trade.provider
    if not isinstance(buyer, ZeroIntelligenceBuyerView) or not isinstance(seller, ZeroIntelligenceSellerView):
        return
    buyer.profit += trade.quantity * (buyer.value - trade.price)
    seller.profit += trade.quantity * (trade.price - seller.value)


@register_model(
    field='economics/markets',
    tags=('market', 'agents', 'double auction', 'zero intelligence'),
    references=('Gooding 2019 ch. 8-12', 'Gode and Sunder 1993'),
    defaults={'until': 20},
)
def double_auction(
    n_buyers: int = 50,
    n_sellers: int = 50,
    *,
    protocol: str = 'order_book',
    rounds: int = 20,
    constrained: bool = True,
    ceiling: float = 200.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Gode and Sunder's double auction of zero-intelligence traders, a step per trading period.

    Every buyer values one unit and every seller can produce one unit at a cost, both drawn uniformly below the price
    ceiling. During a trading period of several rounds, every trader that has not traded yet quotes a random price and
    the market matches the quotes, continuously in a limit order book or at one uniform price in a call auction. With
    the budget constraint (ZI-C) no trader accepts a loss, and the market alone reaches nearly the whole competitive
    surplus; without it (ZI-U), traders make losing trades and the efficiency drops.

    Args:
        n_buyers:
            The number of buyers.

        n_sellers:
            The number of sellers.

        protocol:
            The protocol of the market, `'order_book'` or `'call_auction'`.

        rounds:
            The number of quoting rounds of a trading period.

        constrained:
            Whether traders never quote at a loss.

        ceiling:
            The highest price of the market.

        random_state:
            The seed or generator of the values and the costs.

    Returns:
        system:
            The model, with the groups `buyers` and `sellers`, the market `auction`, and the realised `ledger.surplus`
            and `ledger.efficiency` of every trading period.

    Raises:
        ValueError:
            The protocol is unknown.

    Examples:
        >>> from skcomplex.models.economics.markets import double_auction
        >>> [group.size for group in double_auction(10, 8).elements]
        [10, 8]
    """
    if protocol not in PROTOCOLS:
        error_msg = f'Unknown protocol {protocol!r}. Use one of {PROTOCOLS}.'
        raise ValueError(error_msg)
    rng = np.random.default_rng(random_state)
    values = rng.uniform(0.25 * ceiling, ceiling, n_buyers)
    costs = rng.uniform(0.0, 0.75 * ceiling, n_sellers)
    limit = float(constrained)
    buyers = GroupElement(
        'buyers',
        element=ZeroIntelligenceBuyerElement,
        size=n_buyers,
        values={'value': values, 'ceiling': ceiling, 'constrained': limit},
    )
    sellers = GroupElement(
        'sellers',
        element=ZeroIntelligenceSellerElement,
        size=n_sellers,
        values={'value': costs, 'ceiling': ceiling, 'constrained': limit},
    )
    market = MatchingInteraction(
        'auction',
        requesters='buyers',
        providers='sellers',
        rule=ContinuousMatchingSpec() if protocol == 'order_book' else BatchMatchingSpec(),
        limit=('bid', 'ask'),
        quantity=('demand', 'supply'),
        phase='trade',
        on_match=_book_profit,
    )
    quotes = [
        MapInteraction(label, target=group, state=state, function=function, form='agent', phase='trade')
        for label, group, state, function in (
            ('bids', 'buyers', 'bid', ZeroIntelligenceBuyerElement.get_bid),
            ('asks', 'sellers', 'ask', ZeroIntelligenceSellerElement.get_ask),
        )
    ]
    return System(
        elements=[buyers, sellers],
        interactions=[*quotes, market, AuctionLedgerInteraction('ledger', phase='close')],
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('trade', repeat=rounds), PhaseSpec('close')])),
    )
