"""Define bilateral exchange of goods between agents with Cobb-Douglas preferences, and pairing of neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ....core.laws import transfer


def cobb_douglas(holdings: Sequence[float], weights: Sequence[float]) -> float:
    """Return the logarithm of the Cobb-Douglas utility of a bundle of goods, `sum_i a_i log x_i`.

    Args:
        holdings:
            The holding of every good.

        weights:
            The weight of every good, in the order of `holdings`.

    Returns:
        utility:
            The logarithm of the utility, minus infinity when a weighted good runs out.

    Examples:
        >>> from skcomplex.models.economics.realizations import cobb_douglas
        >>> round(cobb_douglas((4.0, 1.0), (0.5, 0.5)), 6)
        0.693147
    """
    total = 0.0
    for holding, share in zip(holdings, weights, strict=True):
        if share == 0:
            continue
        if holding <= 0:
            return -math.inf
        total += share * math.log(holding)
    return total


def marginal_rate(holdings: tuple[float, float], weights: tuple[float, float]) -> float:
    """Return the marginal rate of substitution of a good for a numeraire, the units of numeraire one unit is worth.

    Args:
        holdings:
            The holdings of the good and of the numeraire.

        weights:
            The weights of the good and of the numeraire.

    Returns:
        rate:
            The ratio of the marginal utilities, `(a_good / x_good) / (a_numeraire / x_numeraire)`.

    Examples:
        >>> from skcomplex.models.economics.realizations import marginal_rate
        >>> marginal_rate((2.0, 6.0), (0.5, 0.5))
        3.0
    """
    return (weights[0] / holdings[0]) / (weights[1] / holdings[1])


def _utility(agent: Any, goods: Sequence[str], weights: Sequence[str]) -> float:  # noqa: ANN401
    """Return the logarithm of the Cobb-Douglas utility of an agent from its states."""
    return cobb_douglas([agent.state(name) for name in goods], [agent.state(name) for name in weights])


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
    if not all(float(agent.state(name)) > 0 for agent in (first, second) for name in names):
        return None
    rates = [
        marginal_rate(
            (float(agent.state(good)), float(agent.state(numeraire))),
            (float(agent.state(names[0])), float(agent.state(names[1]))),
        )
        for agent in (first, second)
    ]
    if math.isclose(rates[0], rates[1]):
        return None
    buyer, seller = (first, second) if rates[0] > rates[1] else (second, first)
    price = math.sqrt(rates[0] * rates[1])
    payment = price * quantity
    if float(seller.state(good)) <= quantity or float(buyer.state(numeraire)) <= payment:
        return None
    before = (_utility(buyer, goods, weights), _utility(seller, goods, weights))
    transfer(seller, buyer, quantity, good)
    transfer(buyer, seller, payment, numeraire)
    if _utility(buyer, goods, weights) > before[0] and _utility(seller, goods, weights) > before[1]:
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
            The first agent view, with a state for the holding and for the weight of every good.

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
        >>> from skcomplex.models.economics.realizations import trade_goods
        >>> class TraderAgent:
        ...     def __init__(self, states):
        ...         self.states = states
        ...     def state(self, name):
        ...         return self.states[name]
        ...     def set_state(self, name, value):
        ...         self.states[name] = value
        >>> first = TraderAgent({'x': 9.0, 'y': 1.0, 'a': 0.5, 'b': 0.5})
        >>> second = TraderAgent({'x': 1.0, 'y': 9.0, 'a': 0.5, 'b': 0.5})
        >>> trades = trade_goods(first, second, ('x', 'y'), ('a', 'b'), quantity=0.5)
        >>> first.state('x') + second.state('x'), len(trades) > 0
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


def neighbor_pairing(
    left: Sequence[Any],
    right: Sequence[Any],
    rng: np.random.Generator,
    capacity: int,
) -> list[tuple[Any, Any]]:
    """Pairing of every agent with a random unpaired neighbour, a pairing of the core's matching law.

    The agents of the first group, in random order, each pick one agent of the second group at random among those in
    the neighbouring cells or nodes that are not yet paired, so trading partners meet only locally.

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

    Examples:
        >>> import numpy as np
        >>> from skcomplex.models.economics.realizations import neighbor_pairing
        >>> neighbor_pairing([], [], np.random.default_rng(0), 1)
        []
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
