"""Order-driven market of noise traders who imitate the expectation of a guru, on a network rewired every day."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import FeedbackInteraction
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, RoleSpec
from ....core.types import Array
from ._guru_trader import GuruTraderView

PRICE, QUANTITY, TRADER = 0, 1, 2


TINY = 1e-12


class GuruMarketInteraction(FeedbackInteraction[SystemView]):
    """Order-driven market of noise traders who imitate the expectation of a guru, on a network rewired every day.

    Every step is a trading day. First every trader picks a random trader `j` and moves its link from the trader `k`
    it follows to `j` with probability `1 / (1 + exp(-beta (f_j - f_k)))`, where the fitness `f` is the wealth relative
    to the richest trader (Eqs. 3.4 and 3.5 of Caiani et al. 2016). The guru is the trader of the largest in-degree of
    the day before, and the intensity of choice `beta` grows by `beta_step` every day the guru survives and returns to
    1 when it is replaced. Every trader draws its expected return `sigma_i eps` (Eq. 3.3) and adopts the one of the
    trader it follows, except the guru.

    Then `tau` intraday operations run, each by a trader drawn at random. With the expected price
    `p_ref exp(sqrt(tau) r)` (Eq. 3.1), a trader expecting a rise bids `U(p (1 - gamma1), p_hat)` for a random share of
    its free cash, and a trader expecting a fall asks `U(p_hat, p (1 + gamma2))` for a random share of its free stocks
    (Eq. 3.2 and Table 3.1). An order that crosses the book is a market order that trades with the best quotes by price
    and time priority, and what is left rests as a limit order, whose cash or stocks stay tied in the book until it
    trades or expires at the end of the day. The price is the last traded price, the mid quote when an operation trades
    nothing, and the previous price when the book is empty.

    Args:
        label:
            The label of the market.

        rule:
            Unused, since the market overrides `act`.

        phase:
            The phase the market acts in, or `None` for the default phase.

        traders:
            The path of the group of traders.

        tau:
            The intraday operations of a day.

        beta_step:
            The growth of the intensity of choice every day the guru survives.

        opening:
            The price before the first day.

    Examples:
        >>> from skcomplex.systems.economics.finance import GuruMarketInteraction
        >>> GuruMarketInteraction('market', traders='traders').reports[:2]
        ('price', 'log_return')
    """

    reports: ClassVar[tuple[str, ...]] = ('price', 'log_return', 'volume', 'guru', 'beta', 'guru_followers')

    def __init__(
        self,
        label: str,
        *,
        rule: None = None,
        phase: str | None = None,
        traders: Annotated[str, RoleSpec()],
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        beta_step: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
        opening: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.traders = traders
        self.tau = tau
        self.beta_step = beta_step
        self.opening = opening

    @staticmethod
    def _stored(system: SystemView, name: str, default: float) -> float:
        """Return a number the market stored in an earlier step, or the default before it is stored."""
        value = system.storage.get(name, default)
        return float(value) if isinstance(value, int | float | np.number) else default

    @staticmethod
    def _rewire(
        follows: Array,
        fitness: Array,
        beta: float,
        rng: np.random.Generator,
    ) -> Array:
        """Return the links after every trader considers a random other trader by the logit of their fitnesses."""
        size = len(follows)
        candidates = rng.integers(size - 1, size=size)
        candidates += candidates >= np.arange(size)
        with np.errstate(over='ignore'):
            probability = 1 / (1 + np.exp(-beta * (fitness[candidates] - fitness[follows])))
        return np.where(rng.random(size) < probability, candidates, follows)

    @staticmethod
    def _fill(
        book: list[list[float]],
        *,
        trader: int,
        limit: float,
        size: float,
        buying: bool,
        holdings: dict[str, Array],
    ) -> tuple[float, float, float]:
        """Trade an incoming order against the opposite side of the book, and return what is left and the last price."""
        volume, last = 0.0, math.nan
        while size > 0 and book and ((limit >= book[0][PRICE]) if buying else (limit <= book[0][PRICE])):
            quote = book[0]
            quantity = min(size, quote[QUANTITY])
            if buying:
                quantity = min(quantity, (holdings['cash'][trader] - holdings['tied_cash'][trader]) / quote[PRICE])
            if quantity <= TINY:
                break
            other = int(quote[TRADER])
            value = quantity * quote[PRICE]
            buyer, seller = (trader, other) if buying else (other, trader)
            holdings['cash'][buyer] -= value
            holdings['cash'][seller] += value
            holdings['stocks'][buyer] += quantity
            holdings['stocks'][seller] -= quantity
            if buying:
                holdings['tied_stocks'][other] -= quantity
            else:
                holdings['tied_cash'][other] -= value
            quote[QUANTITY] -= quantity
            if quote[QUANTITY] <= TINY:
                book.pop(0)
            size -= quantity
            volume += quantity
            last = quote[PRICE]
        return size, volume, last

    def _trade_day(
        self,
        system: SystemView,
        holdings: dict[str, Array],
        expectation: Array,
        price: float,
    ) -> tuple[float, float]:
        """Run the intraday operations of a day, and return the closing price and the volume."""
        rng, tau = system.rng, int(system.parameter('tau'))
        reference, volume = price, 0.0
        bids: list[list[float]] = []
        asks: list[list[float]] = []
        for _ in range(tau):
            trader = int(rng.integers(len(expectation)))
            expected = reference * math.exp(math.sqrt(tau) * expectation[trader])
            buying = expectation[trader] > 0
            if buying:
                low = price * (1 - rng.random())
                limit = low + (expected - low) * rng.random()
                free = holdings['cash'][trader] - holdings['tied_cash'][trader]
                crossing = bool(asks) and limit >= asks[0][PRICE]
                size = rng.random() * free / (asks[0][PRICE] if crossing else limit)
                left, traded, last = self._fill(
                    asks, trader=trader, limit=limit, size=size, buying=True, holdings=holdings,
                )
                free = holdings['cash'][trader] - holdings['tied_cash'][trader]
                left = min(left, free / limit)
            else:
                high = price * (1 + rng.random())
                limit = expected + (high - expected) * rng.random()
                size = rng.random() * (holdings['stocks'][trader] - holdings['tied_stocks'][trader])
                left, traded, last = self._fill(
                    bids, trader=trader, limit=limit, size=size, buying=False, holdings=holdings,
                )
            volume += traded
            if left > TINY:
                side = bids if buying else asks
                side.append([limit, left, float(trader)])
                side.sort(key=(lambda quote: -quote[PRICE]) if buying else (lambda quote: quote[PRICE]))
                holdings['tied_cash' if buying else 'tied_stocks'][trader] += left * limit if buying else left
            if math.isfinite(last):
                price = last
            elif bids and asks:
                price = (bids[0][PRICE] + asks[0][PRICE]) / 2
        return price, volume

    def act(self, system: SystemView) -> None:
        """Rewire the network, form the expectations, and run the trading day.

        Args:
            system:
                The system view of the market.
        """
        traders = sorted(system.agents('traders', GuruTraderView), key=lambda trader: trader.id)
        rng, store = system.rng, system.storage
        price = self._stored(system, 'price', system.parameter('opening'))
        beta = self._stored(system, 'beta', 1.0)
        cash = np.array([trader.cash for trader in traders])
        stocks = np.array([trader.stocks for trader in traders])
        follows = np.array([int(trader.follows) for trader in traders])
        sigma = np.array([trader.sigma for trader in traders])
        guru = int(np.argmax(np.bincount(follows, minlength=len(traders))))
        beta = beta + system.parameter('beta_step') if guru == self._stored(system, 'guru', -1.0) else 1.0
        wealth = cash + price * stocks
        follows = self._rewire(follows, wealth / wealth.max(), beta, rng)
        own = sigma * rng.normal(size=len(traders))
        expectation = own[follows]
        expectation[guru] = own[guru]
        holdings = {
            'cash': cash,
            'stocks': stocks,
            'tied_cash': np.zeros_like(cash),
            'tied_stocks': np.zeros_like(cash),
        }
        closing, volume = self._trade_day(system, holdings, expectation, price)
        followers = np.bincount(follows.astype(int), minlength=len(traders))
        store.update(price=closing, beta=beta, guru=guru)
        for index, trader in enumerate(traders):
            trader.cash, trader.stocks = float(cash[index]), float(stocks[index])
            trader.wealth = float(cash[index] + closing * stocks[index])
            trader.expectation, trader.follows = float(expectation[index]), float(follows[index])
            trader.followers = float(followers[index])
        system.report('price', closing)
        system.report('log_return', math.log(closing / price))
        system.report('volume', volume)
        system.report('guru', float(guru))
        system.report('beta', beta)
        system.report('guru_followers', float(followers.max()))
