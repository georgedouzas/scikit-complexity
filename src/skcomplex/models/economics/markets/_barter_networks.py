"""Bilateral two-good barter on trade networks: complete, isolated, linked, and small-world groups of traders."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt

from ....core.backends import AgentView, ModelView
from ....core.base import Domain, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction, Match, matches
from ....core.laws import Law, Rule, Symbols, transfer
from ....core.runtime import register_model
from ....core.space import DiscreteTime, NetworkSpace
from ....core.system import Observable, System

TOPOLOGIES = ('complete', 'isolated', 'linked', 'small_world')


def _log_spread(rates: npt.ArrayLike) -> float:
    """Return the standard deviation of the logarithms of the marginal rates of substitution."""
    return float(np.std(np.log(np.asarray(rates, dtype=float))))


def _utility(agent: Any) -> float:  # noqa: ANN401
    """Return the Cobb-Douglas utility of the bundle of a trader."""
    return float(agent.good1 * agent.good2)


def _best_neighbours(
    left: Sequence[Any],
    right: Sequence[Any],
    rng: np.random.Generator,
    capacity: int,
) -> list[tuple[Any, Any]]:
    """Return every trader, in random order, paired with the neighbour whose rate of substitution differs the most."""
    pairs = []
    for index in rng.permutation(len(left)):
        trader = left[int(index)]
        rate = math.log(trader.good2 / trader.good1)
        gaps = [(abs(math.log(other.good2 / other.good1) - rate), other) for other in trader.neighbors()]
        if gaps:
            pairs.append((trader, max(gaps, key=lambda item: item[0])[1]))
    return pairs


def _barter(first: AgentView, second: AgentView, units: int) -> int:
    """Trade single units of the first good at the bundle price while both traders gain, and return the units."""
    buyer, seller = (first, second) if first.good2 * second.good1 > second.good2 * first.good1 else (second, first)
    price = (buyer.good2 + seller.good2) / (buyer.good1 + seller.good1)
    traded = 0
    while traded < units:
        buyer_gains = (buyer.good1 + 1) * (buyer.good2 - price) > buyer.good1 * buyer.good2
        seller_gains = (seller.good1 - 1) * (seller.good2 + price) > seller.good1 * seller.good2
        if not (buyer_gains and seller_gains):
            break
        transfer(seller, buyer, 1.0, 'good1')
        transfer(buyer, seller, price, 'good2')
        traded += 1
    return traded


def _trade_network(topology: str, n_agents: int, n_groups: int, shortcuts: int, rng: np.random.Generator) -> nx.Graph:
    """Return the trade network of one of Wilhite's four topologies, with nodes numbered from zero."""
    if topology not in TOPOLOGIES:
        error_msg = f'Unknown topology {topology!r}. Use one of {TOPOLOGIES}.'
        raise ValueError(error_msg)
    if topology == 'complete':
        return nx.complete_graph(n_agents)
    groups = np.array_split(np.arange(n_agents), n_groups)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_agents))
    for members in groups:
        graph.add_edges_from((int(a), int(b)) for index, a in enumerate(members) for b in members[index + 1 :])
    if topology == 'isolated':
        return graph
    for index, members in enumerate(groups):
        following = groups[(index + 1) % n_groups]
        graph.add_edges_from((int(members[-1]), int(other)) for other in following if other != members[-1])
    if topology == 'linked':
        return graph
    for _ in range(shortcuts):
        first, second = rng.choice(n_groups, size=2, replace=False)
        graph.add_edge(int(rng.choice(groups[first])), int(rng.choice(groups[second])))
    return graph


class BarterTrader(Element):
    """Trader with a bundle of two goods and the Cobb-Douglas utility `good1 good2`, who records its rate.

    Its marginal rate of substitution `good2 / good1` is the price of the first good, in units of the second, at
    which it is indifferent to trade. Every step it records the rate of its current bundle.

    Args:
        label:
            The label of the agent or group.

        good1:
            The holding of the first good.

        good2:
            The holding of the second good.

        mrs:
            The marginal rate of substitution of the bundle.

        links:
            The number of trading partners the trader can search.

    Examples:
        >>> from skcomplex.models.economics.markets import BarterTrader
        >>> BarterTrader.group('traders', 4).size
        4
    """

    good1: float = state(default=10.0, domain=Domain.POSITIVE)
    good2: float = state(default=10.0, domain=Domain.POSITIVE)
    mrs: float = state(default=1.0, domain=Domain.POSITIVE)
    links: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule that records the rate of substitution.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.appraise)]

    def appraise(self, agent: AgentView, model: ModelView) -> None:
        """Record the marginal rate of substitution of the bundle.

        Args:
            agent:
                The trader.

            model:
                The model.
        """
        agent.mrs = agent.good2 / agent.good1


class BilateralBarter(Interaction):
    """Barter in which every trader searches its neighbours and trades with the one whose rate differs the most.

    Each pair trades single units of the first good at the price `(g2 + g2') / (g1 + g1')` of their pooled bundles,
    which lies between their rates, while the trade raises the utility of both. The reporters `trades` and
    `searches` count the units traded and the neighbours searched in a step.

    Args:
        label:
            The label of the interaction.

        traders:
            The path of the group of traders.

        units:
            The most units one pair trades in a step.

    Examples:
        >>> from skcomplex.models.economics.markets import BilateralBarter
        >>> BilateralBarter('barter', traders='traders').units
        100
    """

    reports: ClassVar[tuple[str, ...]] = ('trades', 'searches')
    traders: str = role()
    units: int = parameter(default=100, domain=Domain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the matching of every trader with its best neighbour and the rule that trades along the pairs.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        laws: list[Any] = [Match(self.traders, pairing=_best_neighbours, label='partners'), Rule(self.trade)]
        return laws

    def trade(self, model: Any) -> None:  # noqa: ANN401
        """Trade along every pair of the step and report the units traded and the neighbours searched.

        Args:
            model:
                The model view of the interaction.
        """
        units = int(model.parameter('units'))
        traded = sum(_barter(first, second, units) for first, second in matches(model, 'partners'))
        model.report('trades', float(traded))
        model.report('searches', float(sum(trader.links for trader in model.agents(self.traders))))


@register_model(
    field='economics/markets',
    tags=('barter', 'trade', 'network', 'small world', 'agents'),
    references=('Wilhite 2001', 'Gooding 2019'),
    defaults={'until': 60},
    observables=lambda system: [
        Observable('dispersion', group='traders', variable='mrs', reduce=_log_spread),
        Observable('utility', _utility, group='traders'),
    ],
)
def wilhite_trade_networks(
    n_agents: int = 500,
    *,
    topology: str = 'small_world',
    n_groups: int = 10,
    shortcuts: int = 5,
    endowment: float = 100.0,
    units: int = 100,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Wilhite's bilateral barter of two goods among traders on one of four trade networks.

    Traders start with random bundles of two goods and the utility `good1 good2`. Every step, each trader searches
    its neighbours and trades with the one whose marginal rate of substitution differs the most, single units at a
    price between their rates, while both gain. Trade drives the rates of every connected set of traders towards one
    price. On the complete network every trader searches every other, so it needs the most searches, and traders crowd
    onto the most extreme partners, so prices converge slowly. Isolated groups settle on different prices. Groups
    chained by shared end traders, or a small world with some long links, bring the prices together with far fewer
    searches.

    Args:
        n_agents:
            The number of traders.

        topology:
            `'complete'`, `'isolated'` groups, groups `'linked'` in a ring by shared end traders, or `'small_world'`,
            the linked groups with random long links.

        n_groups:
            The number of groups of the isolated, linked, and small-world networks.

        shortcuts:
            The number of random long links between groups of the small world.

        endowment:
            The largest initial holding of each good, drawn uniformly from 1 to it.

        units:
            The most units one pair trades in a step.

        random_state:
            The seed or generator of the bundles and the long links.

    Returns:
        system:
            The model, with the group `traders` on the network, the reporters `barter.trades` and `barter.searches`,
            and the observables `dispersion`, the spread of the logarithms of the rates, and `utility`, the mean
            utility.

    Raises:
        ValueError:
            The topology is unknown.

    Examples:
        >>> from skcomplex.models.economics.markets import wilhite_trade_networks
        >>> system = wilhite_trade_networks(40, topology='isolated', n_groups=4)
        >>> system.elements[0].size
        40
    """
    rng = np.random.default_rng(random_state)
    graph = _trade_network(topology, n_agents, n_groups, shortcuts, rng)
    goods = rng.integers(1, int(endowment) + 1, size=(2, n_agents)).astype(float)
    traders = BarterTrader.group(
        'traders',
        n_agents,
        location=list(range(n_agents)),
        good1=goods[0],
        good2=goods[1],
        mrs=goods[1] / goods[0],
        links=np.array([graph.degree(node) for node in range(n_agents)], dtype=float),
    )
    return System(
        components=[traders],
        interactions=[BilateralBarter('barter', traders='traders', units=units)],
        space=NetworkSpace(graph),
        time=DiscreteTime(),
    )
