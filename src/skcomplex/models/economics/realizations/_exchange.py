"""Define bilateral exchange of goods between agents with Cobb-Douglas preferences, and pairing of neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ....core.laws import transfer


def cobb_douglas(agent: Any, goods: Sequence[str], weights: Sequence[str]) -> float:  # noqa: ANN401
    """Return the logarithm of the Cobb-Douglas utility of an agent, `sum_i a_i log x_i`.

    Args:
        agent:
            The agent, with a state for the holding of every good and for the weight of every good.

        goods:
            The names of the states that hold the goods.

        weights:
            The names of the states that hold the weights of the goods, in the order of `goods`.

    Returns:
        utility:
            The logarithm of the utility, minus infinity when a weighted good runs out.

    Examples:
        >>> from types import SimpleNamespace
        >>> from skcomplex.models.economics.realizations import cobb_douglas
        >>> agent = SimpleNamespace(x=4.0, y=1.0, a=0.5, b=0.5)
        >>> round(cobb_douglas(agent, ('x', 'y'), ('a', 'b')), 6)
        0.693147
    """
    total = 0.0
    for good, weight in zip(goods, weights, strict=True):
        share, holding = float(getattr(agent, weight)), float(getattr(agent, good))
        if share == 0:
            continue
        if holding <= 0:
            return -math.inf
        total += share * math.log(holding)
    return total


def marginal_rate(agent: Any, good: str, numeraire: str, weights: tuple[str, str]) -> float:  # noqa: ANN401
    """Return the marginal rate of substitution of a good for a numeraire, the units of numeraire one unit is worth.

    Args:
        agent:
            The agent.

        good:
            The name of the state that holds the good.

        numeraire:
            The name of the state that holds the numeraire.

        weights:
            The names of the states that hold the weights of the good and of the numeraire.

    Returns:
        rate:
            The ratio of the marginal utilities, `(a_good / x_good) / (a_numeraire / x_numeraire)`.

    Examples:
        >>> from types import SimpleNamespace
        >>> from skcomplex.models.economics.realizations import marginal_rate
        >>> agent = SimpleNamespace(x=2.0, y=6.0, a=0.5, b=0.5)
        >>> marginal_rate(agent, 'x', 'y', ('a', 'b'))
        3.0
    """
    good_weight, numeraire_weight = (float(getattr(agent, name)) for name in weights)
    return (good_weight / float(getattr(agent, good))) / (numeraire_weight / float(getattr(agent, numeraire)))


def _trade_once(
    first: Any,  # noqa: ANN401
    second: Any,  # noqa: ANN401
    pair: tuple[int, int],
    *,
    goods: Sequence[str],
    weights: Sequence[str],
    quantity: float,
) -> float | None:
    """Trade a quantity of one good for its price in another when both traders gain, and return the price."""
    good, numeraire = goods[pair[0]], goods[pair[1]]
    names = (weights[pair[0]], weights[pair[1]])
    if not all(float(getattr(agent, name)) > 0 for agent in (first, second) for name in names):
        return None
    rates = [marginal_rate(agent, good, numeraire, names) for agent in (first, second)]
    if math.isclose(rates[0], rates[1]):
        return None
    buyer, seller = (first, second) if rates[0] > rates[1] else (second, first)
    price = math.sqrt(rates[0] * rates[1])
    payment = price * quantity
    if float(getattr(seller, good)) <= quantity or float(getattr(buyer, numeraire)) <= payment:
        return None
    before = (cobb_douglas(buyer, goods, weights), cobb_douglas(seller, goods, weights))
    transfer(seller, buyer, quantity, good)
    transfer(buyer, seller, payment, numeraire)
    if cobb_douglas(buyer, goods, weights) > before[0] and cobb_douglas(seller, goods, weights) > before[1]:
        return price
    transfer(buyer, seller, quantity, good)
    transfer(seller, buyer, payment, numeraire)
    return None


def trade_goods(
    first: Any,  # noqa: ANN401
    second: Any,  # noqa: ANN401
    goods: Sequence[str],
    weights: Sequence[str],
    *,
    quantity: float = 0.1,
    limit: int = 1000,
) -> list[tuple[int, int, float]]:
    """Barter small quantities of goods between two agents while both gain, and return the trades.

    For every pair of goods in turn, the agent that values the first good more in units of the second buys `quantity`
    of it from the other at the geometric mean of their marginal rates of substitution, as in the Sugarscape trade of
    Epstein and Axtell (1996). A trade happens only when it raises the Cobb-Douglas utility of both agents, and every
    trade is two transfers, so the total of every good is unchanged. The agents trade until a round over all pairs of
    goods makes no trade, or until `limit` trades.

    Args:
        first:
            The first agent, with a state for the holding and for the weight of every good.

        second:
            The second agent.

        goods:
            The names of the states that hold the goods.

        weights:
            The names of the states that hold the weights of the goods, in the order of `goods`.

        quantity:
            The quantity of the bought good in every trade.

        limit:
            The largest number of trades.

    Returns:
        trades:
            The index of the bought good, the index of the good paid with, and the price of every trade.

    Examples:
        >>> from types import SimpleNamespace
        >>> from skcomplex.models.economics.realizations import trade_goods
        >>> first = SimpleNamespace(x=9.0, y=1.0, a=0.5, b=0.5)
        >>> second = SimpleNamespace(x=1.0, y=9.0, a=0.5, b=0.5)
        >>> trades = trade_goods(first, second, ('x', 'y'), ('a', 'b'), quantity=0.5)
        >>> first.x + second.x, len(trades) > 0
        (10.0, True)
    """
    trades: list[tuple[int, int, float]] = []
    pairs = list(itertools.combinations(range(len(goods)), 2))
    while len(trades) < limit:
        traded = False
        for pair in pairs:
            price = _trade_once(first, second, pair, goods=goods, weights=weights, quantity=quantity)
            if price is not None:
                trades.append((pair[0], pair[1], price))
                traded = True
        if not traded:
            break
    return trades


@dataclass(frozen=True)
class NeighbourPairing:
    """Pairing of every agent with a random unpaired neighbour, a pairing of the core's matching law.

    The agents of the first group, in random order, each pick one agent of the second group at random among those in
    the neighbouring cells or nodes that are not yet paired, so trading partners meet only locally.

    Examples:
        >>> from skcomplex.models.economics.realizations import NeighbourPairing
        >>> NeighbourPairing()
        NeighbourPairing()
    """

    def __call__(
        self,
        left: Sequence[Any],
        right: Sequence[Any],
        rng: np.random.Generator,
        capacity: int,
    ) -> list[tuple[Any, Any]]:
        """Return the pairs of neighbours.

        Args:
            left:
                The agents of the first group.

            right:
                The agents of the second group, the same sequence as `left` for pairs within one group.

            rng:
                The random generator.

            capacity:
                The largest number of pairs of every agent.

        Returns:
            pairs:
                The pairs of neighbouring agents.
        """
        candidates = {agent.id for agent in right}
        used: dict[int, int] = {}
        pairs: list[tuple[Any, Any]] = []
        for index in rng.permutation(len(left)):
            agent = left[int(index)]
            if used.get(agent.id, 0) >= capacity:
                continue
            free = [
                other
                for other in agent.neighbors()
                if other.id in candidates and other.id != agent.id and used.get(other.id, 0) < capacity
            ]
            if not free:
                continue
            partner = free[int(rng.integers(len(free)))]
            pairs.append((agent, partner))
            used[agent.id] = used.get(agent.id, 0) + 1
            used[partner.id] = used.get(partner.id, 0) + 1
        return pairs
