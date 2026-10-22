"""Define markets, interactions that match two agent groups with a protocol and settle their trades."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import heapq
import itertools
import math
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
from matching.games import HospitalResident

from ....core.base import role
from ....core.interactions import Interaction
from ....core.laws import Law, Rule, Symbols

type Preference = Callable[[Any, Any], float]

PRICINGS = ('midpoint', 'bid', 'ask')
ORDERS = ('random', 'bid', 'sequential')


@dataclass
class Order:
    """Order of one participant of a market during one clearing.

    Args:
        agent:
            The agent view of the participant.

        limit:
            The highest price a buyer pays or the lowest price a seller accepts.

        quantity:
            The quantity still wanted or offered.

        budget:
            The money a buyer can still spend, infinite without a money state.
    """

    agent: Any
    limit: float
    quantity: float
    budget: float = math.inf

    def affordable(self, price: float) -> float:
        """Return the quantity the participant can still trade at a price."""
        if price <= 0:
            return self.quantity
        return max(0.0, min(self.quantity, self.budget / price))


@dataclass(frozen=True)
class Trade:
    """Trade between a buyer and a seller.

    Args:
        buyer:
            The agent view of the buyer.

        seller:
            The agent view of the seller.

        quantity:
            The traded quantity.

        price:
            The price per unit.
    """

    buyer: Any
    seller: Any
    quantity: float
    price: float

    def __iter__(self) -> Iterator[Any]:
        """Return the buyer and the seller, so a trade is a pair of the matching law."""
        return iter((self.buyer, self.seller))


def _book(trades: list[Trade], buyer: Order, seller: Order, price: float) -> None:
    """Record a trade of the largest quantity both sides can trade at a price and update both orders."""
    quantity = min(buyer.affordable(price), seller.quantity)
    if quantity <= 0:
        return
    buyer.quantity -= quantity
    buyer.budget -= quantity * price
    seller.quantity -= quantity
    trades.append(Trade(buyer.agent, seller.agent, quantity, price))


def _price(buyer: Order, seller: Order, pricing: str) -> float:
    """Return the price of a bilateral trade under a pricing rule."""
    if pricing == 'bid':
        return buyer.limit
    if pricing == 'ask':
        return seller.limit
    return (buyer.limit + seller.limit) / 2


def _check(value: str, allowed: tuple[str, ...], name: str) -> None:
    """Raise an error for an option outside the allowed ones."""
    if value not in allowed:
        error_msg = f'Unknown {name} {value!r}. Use one of {allowed}.'
        raise ValueError(error_msg)


class Protocol:
    """Base class of market protocols, which match buyers with sellers and return the trades.

    Subclasses implement `match(buyers, sellers, model)`, which receives the orders of the participants with something
    to trade and returns trades whose quantities respect the orders' quantities and budgets. A protocol is also a
    pairing of the core's matching law: called with the buyers and sellers, it reads their orders from the states `bid`,
    `demand`, and `money` of buyers and `ask` and `supply` of sellers, and returns the trades, which unpack as pairs of
    buyer and seller.
    """

    def __call__(
        self,
        left: Sequence[Any],
        right: Sequence[Any],
        rng: np.random.Generator,
        capacity: int,
    ) -> list[Any]:
        """Return the trades between buyers and sellers, as a pairing of the matching law.

        Args:
            left:
                The buyers.

            right:
                The sellers.

            rng:
                The random generator.

            capacity:
                The largest number of trades of every participant, which protocols do not limit.

        Returns:
            trades:
                The trades.
        """
        buyers = [
            Order(agent, _read(agent, 'bid', math.inf), _read(agent, 'demand', 1.0), _read(agent, 'money', math.inf))
            for agent in left
        ]
        sellers = [Order(agent, _read(agent, 'ask', 0.0), _read(agent, 'supply', 1.0)) for agent in right]
        buyers = [order for order in buyers if order.quantity > 0]
        sellers = [order for order in sellers if order.quantity > 0]
        model = SimpleNamespace(rng=rng, storage={})
        return self.match(buyers, sellers, model) if buyers and sellers else []

    def match(self, buyers: list[Order], sellers: list[Order], model: Any) -> list[Trade]:  # noqa: ANN401
        """Return the trades between buyers and sellers.

        Args:
            buyers:
                The orders of the buyers.

            sellers:
                The orders of the sellers.

            model:
                The model view of the market.

        Returns:
            trades:
                The trades.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class BilateralSearch(Protocol):
    """Random bilateral search: buyers in random order meet random sellers and trade when their limits cross.

    Every buyer makes up to `tries` meetings with sellers that still offer something, and trades at the midpoint of
    the two limits, or at the bid or the ask, the protocol of decentralised exchange of Hamill and Gilbert (2016 ch. 5).

    Args:
        tries:
            The number of sellers every buyer meets.

        pricing:
            The price of a trade, `'midpoint'`, `'bid'`, or `'ask'`.

    Examples:
        >>> from skcomplex.models.economics.realizations import BilateralSearch
        >>> BilateralSearch(tries=3).tries
        3
    """

    tries: int = 1
    pricing: str = 'midpoint'

    def __post_init__(self) -> None:
        """Check the pricing rule."""
        _check(self.pricing, PRICINGS, 'pricing')

    def match(self, buyers: list[Order], sellers: list[Order], model: Any) -> list[Trade]:  # noqa: ANN401
        """Return the trades of random meetings."""
        trades: list[Trade] = []
        for index in model.rng.permutation(len(buyers)):
            buyer = buyers[index]
            for _ in range(self.tries):
                available = [seller for seller in sellers if seller.quantity > 0]
                if buyer.quantity <= 0 or not available:
                    break
                seller = available[int(model.rng.integers(len(available)))]
                if seller.limit <= buyer.limit:
                    _book(trades, buyer, seller, _price(buyer, seller, self.pricing))
        return trades


@dataclass(frozen=True)
class SampleAndSwitch(Protocol):
    """Partner choice that samples sellers and switches to a cheaper one with some probability.

    The protocol follows chapter 2 of Caiani et al. (2016). Every buyer, in random order, samples `sample` sellers that
    still offer something. When the cheapest of them asks less than its current partner, the buyer switches with
    probability `1 - exp(intensity * (p_new - p_old) / p_new)`, and otherwise stays. It buys from its partner at the
    partner's ask, and the buyer's state `partner`, when it has one, holds the id of the partner across steps.

    Args:
        sample:
            The number of sellers every buyer samples.

        intensity:
            The intensity of switching.

        partner:
            The name of the buyer state that holds the id of its partner.

    Examples:
        >>> from skcomplex.models.economics.realizations import SampleAndSwitch
        >>> SampleAndSwitch(sample=2, intensity=4.0).intensity
        4.0
    """

    sample: int = 3
    intensity: float = 1.0
    partner: str = 'partner'

    def match(self, buyers: list[Order], sellers: list[Order], model: Any) -> list[Trade]:  # noqa: ANN401
        """Return the trades of every buyer with its kept or new partner."""
        trades: list[Trade] = []
        by_id = {seller.agent.id: seller for seller in sellers}
        for index in model.rng.permutation(len(buyers)):
            buyer = buyers[index]
            available = [seller for seller in sellers if seller.quantity > 0]
            if not available:
                break
            picks = model.rng.choice(len(available), size=min(self.sample, len(available)), replace=False)
            best = min((available[int(pick)] for pick in picks), key=lambda seller: seller.limit)
            current = by_id.get(int(_read(buyer.agent, self.partner, -1)))
            chosen = best
            if current is not None and current.quantity > 0 and best.limit >= current.limit:
                chosen = current
            elif current is not None and current.quantity > 0 and best.limit > 0:
                switching = 1 - math.exp(self.intensity * (best.limit - current.limit) / best.limit)
                chosen = best if model.rng.random() < switching else current
            _write(buyer.agent, self.partner, chosen.agent.id)
            if chosen.limit <= buyer.limit:
                _book(trades, buyer, chosen, chosen.limit)
        return trades


@dataclass(frozen=True)
class RationedQueue(Protocol):
    """Queue of buyers served by the cheapest sellers until the goods run out, so the last buyers are rationed.

    Buyers queue in random order, by decreasing bid, or in order of creation, and each buys at the ask of the cheapest
    sellers whose ask it accepts.

    Args:
        order:
            The order of the queue, `'random'`, `'bid'`, or `'sequential'`.

    Examples:
        >>> from skcomplex.models.economics.realizations import RationedQueue
        >>> RationedQueue(order='bid').order
        'bid'
    """

    order: str = 'random'

    def __post_init__(self) -> None:
        """Check the order of the queue."""
        _check(self.order, ORDERS, 'order')

    def match(self, buyers: list[Order], sellers: list[Order], model: Any) -> list[Trade]:  # noqa: ANN401
        """Return the trades of the queue."""
        queue = _ordered(buyers, self.order, model)
        stock = sorted(sellers, key=lambda seller: seller.limit)
        trades: list[Trade] = []
        for buyer in queue:
            for seller in stock:
                if buyer.quantity <= 0 or seller.limit > buyer.limit:
                    break
                if seller.quantity > 0:
                    _book(trades, buyer, seller, seller.limit)
        return trades


def _ordered(orders: list[Order], order: str, model: Any) -> list[Order]:  # noqa: ANN401
    """Return orders in random order, by decreasing limit, or in order of creation."""
    if order == 'bid':
        return sorted(orders, key=lambda item: (-item.limit, item.agent.id))
    if order == 'sequential':
        return sorted(orders, key=lambda item: item.agent.id)
    return [orders[index] for index in model.rng.permutation(len(orders))]


def _clear_at(buyers: list[Order], sellers: list[Order], price: float, model: Any) -> list[Trade]:  # noqa: ANN401
    """Return the trades at one price between the willing buyers and sellers in random order."""
    trades: list[Trade] = []
    willing_buyers = _ordered([buyer for buyer in buyers if buyer.limit >= price], 'random', model)
    willing_sellers = _ordered([seller for seller in sellers if seller.limit <= price], 'random', model)
    pairs = iter(willing_sellers)
    seller = next(pairs, None)
    for buyer in willing_buyers:
        while seller is not None and buyer.affordable(price) > 0:
            _book(trades, buyer, seller, price)
            if seller.quantity <= 0:
                seller = next(pairs, None)
    return trades


@dataclass(frozen=True)
class Walrasian(Protocol):
    """Walrasian auctioneer: tâtonnement to the price that balances demand and supply, then trade at that price.

    Starting from the last clearing price, or the median of the limits, the auctioneer moves the price by
    `p <- p (1 + adjustment (D - S) / (D + S))` until the relative excess demand falls below `tolerance`, where `D` is
    the demand of buyers whose bid is at least `p` and `S` the supply of sellers whose ask is at most `p`. The long side
    is rationed in random order (Hamill and Gilbert 2016 ch. 5).

    Args:
        adjustment:
            The speed of the price adjustment.

        iterations:
            The largest number of adjustments per clearing.

        tolerance:
            The relative excess demand at which the adjustment stops.

    Examples:
        >>> from skcomplex.models.economics.realizations import Walrasian
        >>> Walrasian(adjustment=0.05).iterations
        500
    """

    adjustment: float = 0.1
    iterations: int = 500
    tolerance: float = 1e-3

    def price(self, buyers: list[Order], sellers: list[Order], start: float) -> float:
        """Return the price the tâtonnement reaches from a start.

        Args:
            buyers:
                The orders of the buyers.

            sellers:
                The orders of the sellers.

            start:
                The starting price.

        Returns:
            price:
                The price.
        """
        price = start
        for _ in range(self.iterations):
            demand = sum(buyer.affordable(price) for buyer in buyers if buyer.limit >= price)
            supply = sum(seller.quantity for seller in sellers if seller.limit <= price)
            total = demand + supply
            if not total or abs(demand - supply) / total < self.tolerance:
                break
            price *= 1 + self.adjustment * (demand - supply) / total
        return price

    def match(self, buyers: list[Order], sellers: list[Order], model: Any) -> list[Trade]:  # noqa: ANN401
        """Return the trades at the price of the tâtonnement."""
        limits = [order.limit for order in buyers + sellers if order.limit > 0]
        if not limits:
            return []
        last = model.storage.get('price', math.nan)
        start = last if math.isfinite(last) and last > 0 else float(np.median(limits))
        price = self.price(buyers, sellers, start)
        model.storage['price'] = price
        return _clear_at(buyers, sellers, price, model)


@dataclass(frozen=True)
class OrderBook(Protocol):
    """Limit order book with price-time priority, cleared continuously as orders arrive.

    Every participant submits one limit order, in random order of arrival or in order of creation. An incoming order
    trades with the best opposite orders, the oldest first among equal prices, at their prices, as long as the prices
    cross, and its remainder rests in the book. The book is emptied at the end of every clearing (Caiani et al. 2016
    ch. 3).

    Args:
        arrival:
            The order of arrival, `'random'` or `'sequential'`.

    Examples:
        >>> from skcomplex.models.economics.realizations import OrderBook
        >>> OrderBook().arrival
        'random'
    """

    arrival: str = 'random'

    def __post_init__(self) -> None:
        """Check the order of arrival."""
        _check(self.arrival, ('random', 'sequential'), 'arrival')

    def match(self, buyers: list[Order], sellers: list[Order], model: Any) -> list[Trade]:  # noqa: ANN401
        """Return the trades of the continuous double auction."""
        orders = [(order, True) for order in buyers] + [(order, False) for order in sellers]
        if self.arrival == 'sequential':
            orders.sort(key=lambda item: item[0].agent.id)
        else:
            orders = [orders[index] for index in model.rng.permutation(len(orders))]
        bids: list[tuple[float, int, Order]] = []
        asks: list[tuple[float, int, Order]] = []
        trades: list[Trade] = []
        for time, (order, is_buy) in enumerate(orders):
            if is_buy:
                while asks and asks[0][0] <= order.limit and order.affordable(asks[0][0]) > 0:
                    resting = asks[0][2]
                    _book(trades, order, resting, resting.limit)
                    if resting.quantity <= 0:
                        heapq.heappop(asks)
                if order.affordable(order.limit) > 0:
                    heapq.heappush(bids, (-order.limit, time, order))
            else:
                while bids and -bids[0][0] >= order.limit and order.quantity > 0:
                    resting = bids[0][2]
                    _book(trades, resting, order, resting.limit)
                    if resting.affordable(resting.limit) <= 0:
                        heapq.heappop(bids)
                if order.quantity > 0:
                    heapq.heappush(asks, (order.limit, time, order))
        return trades


@dataclass(frozen=True)
class CallAuction(Protocol):
    """Call auction: collect every order, then trade the largest volume at one uniform price.

    Bids are sorted by decreasing and asks by increasing limit, and they are paired while the bid covers the ask. The
    price is the midpoint of the last paired bid and ask, and every paired buyer and seller trades at it.

    Examples:
        >>> from skcomplex.models.economics.realizations import CallAuction
        >>> CallAuction()
        CallAuction()
    """

    def match(self, buyers: list[Order], sellers: list[Order], model: Any) -> list[Trade]:  # noqa: ANN401
        """Return the trades at the uniform clearing price."""
        bids = sorted(buyers, key=lambda order: (-order.limit, order.agent.id))
        asks = sorted(sellers, key=lambda order: (order.limit, order.agent.id))
        left = {
            id(order): order.quantity if not math.isfinite(order.budget) else order.affordable(order.limit)
            for order in bids
        }
        offered = {id(order): order.quantity for order in asks}
        pairs, marginal = [], None
        bid_iter, ask_iter = iter(bids), iter(asks)
        bid, ask = next(bid_iter, None), next(ask_iter, None)
        while bid is not None and ask is not None and bid.limit >= ask.limit:
            pairs.append((bid, ask))
            marginal = (bid.limit + ask.limit) / 2
            quantity = min(left[id(bid)], offered[id(ask)])
            left[id(bid)] -= quantity
            offered[id(ask)] -= quantity
            if left[id(bid)] <= 0:
                bid = next(bid_iter, None)
            if offered[id(ask)] <= 0:
                ask = next(ask_iter, None)
        trades: list[Trade] = []
        if marginal is None:
            return trades
        for bid, ask in pairs:
            _book(trades, bid, ask, marginal)
        return trades


def _cheaper(buyer: Any, seller: Any) -> float:  # noqa: ANN401
    """Return the preference of a buyer for a seller, higher for a lower ask."""
    del buyer
    return -float(seller.ask) if _has(seller, 'ask') else 0.0


def _richer(seller: Any, buyer: Any) -> float:  # noqa: ANN401
    """Return the preference of a seller for a buyer, higher for a higher bid."""
    del seller
    return float(buyer.bid) if _has(buyer, 'bid') else 0.0


@dataclass(frozen=True)
class DeferredAcceptance(Protocol):
    """Stable matching of Gale and Shapley, buyers proposing to sellers that hold as many buyers as they offer units.

    Buyers rank the sellers whose ask they accept by `buyer_preference(buyer, seller)`, by default the lower ask, and
    sellers rank the buyers by `seller_preference(seller, buyer)`, by default the higher bid. The buyer-optimal
    stable matching, from the hospital-resident solver of the `matching` library, pairs every buyer with at most one
    seller, and each pair trades one unit at the price of the pricing rule.

    Args:
        buyer_preference:
            The score of a seller for a buyer, higher for preferred sellers, or `None` for the lower ask.

        seller_preference:
            The score of a buyer for a seller, higher for preferred buyers, or `None` for the higher bid.

        pricing:
            The price of a match, `'ask'`, `'bid'`, or `'midpoint'`.

    Examples:
        >>> from skcomplex.models.economics.realizations import DeferredAcceptance
        >>> DeferredAcceptance(pricing='bid').pricing
        'bid'
    """

    buyer_preference: Preference | None = None
    seller_preference: Preference | None = None
    pricing: str = 'ask'

    def __post_init__(self) -> None:
        """Check the pricing rule."""
        _check(self.pricing, PRICINGS, 'pricing')

    def match(self, buyers: list[Order], sellers: list[Order], model: Any) -> list[Trade]:  # noqa: ANN401
        """Return the trades of the stable matching."""
        del model
        rank_seller = self.buyer_preference or _cheaper
        rank_buyer = self.seller_preference or _richer
        capacities = {seller.agent.id: int(seller.quantity) for seller in sellers if int(seller.quantity) > 0}
        by_id = {order.agent.id: order for order in itertools.chain(buyers, sellers)}
        buyer_prefs = {}
        for buyer in buyers:
            acceptable = [seller for seller in sellers if seller.agent.id in capacities and seller.limit <= buyer.limit]
            ranked = sorted((-rank_seller(buyer.agent, seller.agent), seller.agent.id) for seller in acceptable)
            if ranked:
                buyer_prefs[buyer.agent.id] = [identity for _, identity in ranked]
        if not buyer_prefs:
            return []
        seller_prefs = {}
        for seller_id in capacities:
            seller = by_id[seller_id]
            suitors = sorted(
                (-rank_buyer(seller.agent, by_id[buyer_id].agent), buyer_id)
                for buyer_id, ranked in buyer_prefs.items()
                if seller_id in ranked
            )
            seller_prefs[seller_id] = [identity for _, identity in suitors]
        seller_prefs = {key: value for key, value in seller_prefs.items() if value}
        capacities = {key: capacities[key] for key in seller_prefs}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            game = HospitalResident.create_from_dictionaries(buyer_prefs, seller_prefs, capacities, clean=True)
            solution = game.solve(optimal='resident')
        trades: list[Trade] = []
        for hospital, residents in solution.items():
            seller = by_id[hospital.name]
            for resident in residents:
                buyer = by_id[resident.name]
                price = _price(buyer, seller, self.pricing)
                quantity = min(1.0, buyer.affordable(price), seller.quantity)
                if quantity > 0:
                    buyer.quantity -= quantity
                    buyer.budget -= quantity * price
                    seller.quantity -= quantity
                    trades.append(Trade(buyer.agent, seller.agent, quantity, price))
        return trades


def _has(agent: Any, name: str | None) -> bool:  # noqa: ANN401
    """Check whether an agent has a state or parameter."""
    if name is None:
        return False
    try:
        getattr(agent, name)
    except AttributeError:
        return False
    return True


def _read(agent: Any, name: str | None, default: float) -> float:  # noqa: ANN401
    """Return a state or parameter of an agent, or a default when it has none."""
    return float(getattr(agent, name)) if name is not None and _has(agent, name) else default


def _write(agent: Any, name: str | None, value: float) -> None:  # noqa: ANN401
    """Set a state of an agent when it has one."""
    if name is not None and _has(agent, name):
        setattr(agent, name, value)


class Market(Interaction):
    """Market between two agent groups, matched by a protocol and settled by transfers between the agents.

    Buyers state the highest price they pay in the state `bid` and the quantity they want in `demand`, and sellers the
    lowest price they accept in `ask` and the quantity they offer in `supply`; a missing quantity counts as one unit.
    Every trade moves `price * quantity` of the state `money` from buyer to seller and `quantity` of the state `goods`
    from seller to buyer, when the agents have those states, and lowers `demand` and `supply`, so it is atomic and
    conserves both. A buyer's money bounds what it can buy. `on_trade(trade, model)` adds any other effect, such as
    recording an employer. The market runs once per round of its phase and reports the volume-weighted `price`, the
    traded `volume`, and the `unmatched` demand and supply left, as the reporters `<market>.price`, `<market>.volume`,
    and `<market>.unmatched`.

    Args:
        label:
            The label of the market.

        buyers:
            The path of the group of buyers.

        sellers:
            The path of the group of sellers.

        protocol:
            The protocol that matches the participants, such as `OrderBook()`.

        phase:
            The phase the market clears in, or `None` for the default phase.

        bid:
            The state of a buyer with its limit price.

        ask:
            The state of a seller with its limit price.

        demand:
            The state of a buyer with the quantity it wants.

        supply:
            The state of a seller with the quantity it offers.

        money:
            The state that pays for trades, or `None` for no payment.

        goods:
            The state that holds the traded goods, or `None` for no delivery.

        on_trade:
            A function of every trade and the model view, run after the transfers, or `None`.

    Examples:
        >>> from skcomplex.models.economics.realizations import Market, OrderBook
        >>> Market('exchange', buyers='traders', sellers='traders', protocol=OrderBook()).reports
        ('price', 'volume', 'unmatched')
    """

    reports: ClassVar[tuple[str, ...]] = ('price', 'volume', 'unmatched')
    buyers: str = role()
    sellers: str = role()
    protocol: Protocol = field(default_factory=BilateralSearch)
    phase: str | None = None
    bid: str = 'bid'
    ask: str = 'ask'
    demand: str = 'demand'
    supply: str = 'supply'
    money: str | None = 'money'
    goods: str | None = 'goods'
    on_trade: Callable[[Trade, Any], None] | None = None

    def laws(self, s: Symbols) -> list[Law]:
        """Return the clearing rule in the market's phase.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.clear, phase=self.phase)]

    def orders(self, model: Any) -> tuple[list[Order], list[Order]]:  # noqa: ANN401
        """Return the orders of the buyers and the sellers with something to trade.

        Args:
            model:
                The model view of the market.

        Returns:
            orders:
                The orders of the buyers and of the sellers.
        """
        buyers = [
            Order(
                agent,
                _read(agent, self.bid, math.inf),
                _read(agent, self.demand, 1.0),
                _read(agent, self.money, math.inf),
            )
            for agent in model.agents('buyers')
        ]
        sellers = [
            Order(agent, _read(agent, self.ask, 0.0), _read(agent, self.supply, 1.0))
            for agent in model.agents('sellers')
        ]
        return [order for order in buyers if order.quantity > 0], [order for order in sellers if order.quantity > 0]

    def settle(self, trade: Trade, model: Any) -> None:  # noqa: ANN401
        """Transfer money and goods of a trade and lower the demand and supply.

        Args:
            trade:
                The trade.

            model:
                The model view of the market.
        """
        amount = trade.quantity * trade.price
        buyer, seller = trade.buyer, trade.seller
        for agent, name, change in (
            (buyer, self.money, -amount),
            (seller, self.money, amount),
            (buyer, self.goods, trade.quantity),
            (seller, self.goods, -trade.quantity),
            (buyer, self.demand, -trade.quantity),
            (seller, self.supply, -trade.quantity),
        ):
            if name is not None and _has(agent, name):
                setattr(agent, name, getattr(agent, name) + change)
        if self.on_trade is not None:
            self.on_trade(trade, model)

    def clear(self, model: Any) -> None:  # noqa: ANN401
        """Match the participants, settle the trades, and report the price, the volume, and the unmatched quantity.

        Args:
            model:
                The model view of the market.
        """
        buyers, sellers = self.orders(model)
        trades = self.protocol.match(buyers, sellers, model) if buyers and sellers else []
        for trade in trades:
            self.settle(trade, model)
        volume = sum(trade.quantity for trade in trades)
        price = sum(trade.quantity * trade.price for trade in trades) / volume if volume else math.nan
        unmatched = sum(order.quantity for order in buyers + sellers)
        model.report('price', price)
        model.report('volume', volume)
        model.report('unmatched', unmatched)


def total_surplus(trades: Sequence[Trade], value: Callable[[Any], float], cost: Callable[[Any], float]) -> float:
    """Return the surplus of trades, the buyers' values minus the sellers' costs of the traded units.

    Args:
        trades:
            The trades.

        value:
            The value of a unit to a buyer, a function of the buyer.

        cost:
            The cost of a unit to a seller, a function of the seller.

    Returns:
        surplus:
            The total surplus.

    Examples:
        >>> from skcomplex.models.economics.realizations import Trade, total_surplus
        >>> total_surplus([Trade('b', 's', 2.0, 5.0)], lambda buyer: 7.0, lambda seller: 4.0)
        6.0
    """
    return float(sum(trade.quantity * (value(trade.buyer) - cost(trade.seller)) for trade in trades))
