"""Bilateral two-good barter on trade networks: complete, isolated, linked, and small-world groups of traders."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import networkx as nx
import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.interactions import MapInteraction
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import System
from ....core.specs import (
    ObservableSpec,
    PartnerSpec,
)
from ..._catalog import register_system
from ._barter_trader import BarterTraderElement
from ._bilateral_barter import BilateralBarterInteraction

TOPOLOGIES = ('complete', 'isolated', 'linked', 'small_world')


GOODS = ('good1', 'good2')


def _log_spread(rates: object) -> float:
    """Return the standard deviation of the logarithms of the marginal rates of substitution."""
    return float(np.std(np.log(np.asarray(rates, dtype=float))))


def _utility(agent: object) -> float:
    """Return the Cobb-Douglas utility of the bundle of a trader."""
    return math.prod(float(getattr(agent, good)) for good in GOODS)


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


@register_system(
    field='economics/markets',
    tags=('barter', 'trade', 'network', 'small world', 'agents'),
    references=('Wilhite 2001', 'Gooding 2019'),
    defaults={'until': 60},
    observables=lambda system: [
        ObservableSpec('dispersion', group='traders', variable='mrs', reduce=_log_spread),
        ObservableSpec('utility', _utility, group='traders'),
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
            The system, with the group `traders` on the network, the reporters `barter.trades` and `barter.searches`,
            and the observables `dispersion`, the spread of the logarithms of the rates, and `utility`, the mean
            utility.

    Raises:
        ValueError:
            The topology is unknown.

    Examples:
        >>> from skcomplex.systems.economics.markets import wilhite_trade_networks
        >>> system = wilhite_trade_networks(40, topology='isolated', n_groups=4)
        >>> system.elements[0].prepare().size_
        40
    """
    rng = np.random.default_rng(random_state)
    graph = _trade_network(topology, n_agents, n_groups, shortcuts, rng)
    goods = rng.integers(1, int(endowment) + 1, size=(2, n_agents)).astype(float)
    traders = GroupElement(
        'traders',
        element=BarterTraderElement,
        size=n_agents,
        values={
            'good1': goods[0],
            'good2': goods[1],
            'mrs': goods[1] / goods[0],
            'links': np.array([graph.degree(node) for node in range(n_agents)], dtype=float),
        },
        location=list(range(n_agents)),
    )
    return System(
        elements=[traders],
        interactions=[
            MapInteraction(
                'appraisal',
                target='traders',
                state='mrs',
                function=BarterTraderElement.get_rate,
                form='agent',
            ),
            BilateralBarterInteraction(
                'barter',
                requesters='traders',
                providers='traders',
                rule=PartnerSpec(choice='score', score=BarterTraderElement.get_rate_gap, clearing='provider'),
                quantity=('quantity', 'partners'),
                units=units,
            ),
        ],
        space=NetworkSpace(graph),
        time=DiscreteTime(),
    )
