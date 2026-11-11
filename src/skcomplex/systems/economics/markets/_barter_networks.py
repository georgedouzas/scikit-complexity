"""Bilateral two-good barter on trade networks.

The networks are complete, isolated, linked, and small-world groups of traders.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Sequence
from typing import Annotated, ClassVar

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction
from ....core.components.laws.utils import transfer
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ...general.matching import MatchingInteraction, MatchSpec, PartnerMatchingSpec, RandomSearchSpec
from ...general.matching.types import MatchingRule

TOPOLOGIES = ('complete', 'isolated', 'linked', 'small_world')
GOODS = ('good1', 'good2')


class BarterTraderView(AgentView):
    """Barter trader as the rules see it.

    It holds its holdings `good1` and `good2`, its rate `mrs`, `links`, and `partners`.
    """

    good1: float
    good2: float
    mrs: float
    links: float
    partners: float


class BarterTraderElement(Element):
    """Trader with a bundle of two goods and the Cobb-Douglas utility `good1 good2`.

    It records its rate. Its marginal rate of substitution `good2 / good1` is the price of the first good, in units of
    the second, at which it is indifferent to trade. Every step it records the rate of its current bundle.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        good1:
            The holding of the first good.

        good2:
            The holding of the second good.

        mrs:
            The marginal rate of substitution of the bundle.

        links:
            The number of trading partners the trader can search.

        partners:
            The number of partners that can match with the trader in a step, without limit.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import BarterTraderElement
        >>> GroupElement('traders', element=BarterTraderElement, size=4).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        good1: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 10.0,
        good2: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 10.0,
        mrs: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        links: Annotated[float, StateSpec(domain=NN)] = 0.0,
        partners: Annotated[float, StateSpec()] = math.inf,
    ) -> None:
        super().__init__(label, visual=visual)
        self.good1 = good1
        self.good2 = good2
        self.mrs = mrs
        self.links = links
        self.partners = partners

    @classmethod
    def _get_view(cls) -> type[BarterTraderView]:
        """Return the view of a trader, with its holdings, rate, and links."""
        return BarterTraderView

    @staticmethod
    def is_partner(trader: object, other: object) -> bool:
        """Return whether another trader is a trading partner of a trader.

        A partner is a neighbour on the trade network.

        Args:
            trader:
                The trader that searches.

            other:
                Another trader.

        Returns:
            partner:
                Whether the other trader is in a node next to the trader's node.
        """
        if not isinstance(trader, BarterTraderView) or not isinstance(other, BarterTraderView):
            return False
        return other.id != trader.id and other.location in trader.get_neighbor_cells()

    @staticmethod
    def get_rate_gap(trader: object, other: object) -> float:
        """Return the gap between the logarithms of the rates of substitution.

        The rates are those of a trader and a trading partner. The gap is minus infinity for a trader that is not a
        partner, so a trader takes a partner whenever it has one.

        Args:
            trader:
                The trader that searches.

            other:
                Another trader.

        Returns:
            gap:
                The absolute difference of the logarithms of the two rates.
        """
        if not BarterTraderElement.is_partner(trader, other):
            return -math.inf
        if not isinstance(trader, BarterTraderView) or not isinstance(other, BarterTraderView):
            return -math.inf
        return abs(math.log(other.good2 / other.good1) - math.log(trader.good2 / trader.good1))

    @staticmethod
    def get_rate(agent: BarterTraderView, system: SystemView) -> float:
        """Return the marginal rate of substitution of the bundle, `good2 / good1`.

        Args:
            agent:
                The trader.

            system:
                The system view of the appraisal.

        Returns:
            mrs:
                The marginal rate of substitution.
        """
        return agent.good2 / agent.good1


class BilateralBarterInteraction(MatchingInteraction[SystemView]):
    """Barter in which every trader trades with the neighbour it matched.

    A greedy matching pairs every trader, in random order, with the neighbour whose marginal rate of substitution
    differs the most in logarithm, and a neighbour takes any number of partners. The score of a trader that is not a
    neighbour is minus infinity, and a trader without neighbours trades with no one. Each pair trades single units of
    the first good at the price `(g2 + g2') / (g1 + g1')` of their pooled bundles, which lies between their rates, while
    the trade raises the utility of both. Besides the reporters of a matching, the reporters `trades` and `searches`
    count the units traded and the neighbours searched in a step.

    Args:
        label:
            The label of the interaction.

        requesters:
            The path of the group of traders that search for a partner.

        providers:
            The path of the group of traders that can be found, the same group.

        rule:
            The spec of the algorithm that pairs the traders, such as `PartnerMatchingSpec('local', value='provider')`.

        reserve:
            The state with the reserve value of both sides, missing for traders, which then accept any partner.

        quantity:
            The states of the partners a requester wants and a provider takes, one and without limit for traders.

        cap:
            The state of a requester with the most value it can give, missing for traders.

        transfers:
            The states every match moves, none for traders, whose trades move the goods.

        phase:
            The phase the barter runs in, or `None` for the default phase.

        on_match:
            A function of every match and the system view, run after the trade, or `None`.

        units:
            The most units one pair trades in a step.

    Examples:
        >>> from skcomplex.systems.economics.markets import BilateralBarterInteraction
        >>> BilateralBarterInteraction('barter', requesters='traders', providers='traders').units
        100
    """

    reports: ClassVar[tuple[str, ...]] = (*MatchingInteraction.reports, 'trades', 'searches')

    def __init__(
        self,
        label: str,
        *,
        requesters: Annotated[str, RoleSpec()],
        providers: Annotated[str, RoleSpec()],
        rule: MatchingRule = RandomSearchSpec(),
        reserve: str | tuple[str, str] = 'reserve',
        quantity: str | tuple[str, str] = 'quantity',
        cap: str = 'cap',
        transfers: Sequence[tuple[str, int]] = (),
        phase: str | None = None,
        on_match: Callable[[MatchSpec, SystemView], None] | None = None,
        units: Annotated[int, ParameterSpec(domain=NN)] = 100,
    ) -> None:
        super().__init__(
            label,
            requesters=requesters,
            providers=providers,
            rule=rule,
            reserve=reserve,
            quantity=quantity,
            cap=cap,
            transfers=transfers,
            phase=phase,
            on_match=on_match,
        )
        self.units = units

    @staticmethod
    def _barter(first: BarterTraderView, second: BarterTraderView, units: int) -> int:
        """Trade single units of the first good at the bundle price while both gain.

        It returns the units.
        """
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

    def _settle(self, match: MatchSpec, system: SystemView) -> None:
        """Trade along a pair of partners, count the units traded, and settle the match."""
        first, second = match.requester, match.provider
        if not isinstance(first, BarterTraderView) or not isinstance(second, BarterTraderView):
            super()._settle(match, system)
            return
        if BarterTraderElement.is_partner(first, second):
            traded = self._barter(first, second, int(system.get_parameter('units')))
            system.storage['trades'] = self._get_trades(system) + traded
        super()._settle(match, system)

    @staticmethod
    def _get_trades(system: SystemView) -> float:
        """Return the units traded so far in the step, kept in the storage of the barter."""
        trades = system.storage.get('trades')
        return float(trades) if isinstance(trades, int | float) else 0.0

    def match(self, system: SystemView) -> None:
        """Pair and trade, and report the units traded and the neighbours searched.

        The report comes besides the matching reporters.

        Args:
            system:
                The system view of the interaction.
        """
        system.storage['trades'] = 0.0
        super().match(system)
        system.report('trades', self._get_trades(system))
        traders = system.get_agents('requesters', BarterTraderView)
        system.report('searches', float(sum(trader.links for trader in traders)))


def _get_utility(agent: object) -> float:
    """Return the Cobb-Douglas utility of the bundle of a trader."""
    return math.prod(float(getattr(agent, good)) for good in GOODS)


def _log_spread(rates: object) -> float:
    """Return the standard deviation of the log marginal rates of substitution."""
    return float(np.std(np.log(np.asarray(rates, dtype=float))))


class WilhiteTradeNetworksSystem(BaseSystem):
    """Wilhite's bilateral barter of two goods among traders on a trade network.

    The network is one of four. Traders start with random bundles of two goods and the utility `good1 good2`. Every
    step, each trader searches its neighbours and trades with the one whose marginal rate of substitution differs the
    most, single units at a price between their rates, while both gain. Trade drives the rates of every connected set of
    traders towards one price. On the complete network every trader searches every other, so it needs the most searches,
    and traders crowd onto the most extreme partners, so prices converge slowly. Isolated groups settle on different
    prices. Groups chained by shared end traders, or a small world with some long links, bring the prices together with
    far fewer searches.

    The system has the group `traders` on the network, the reporters `barter.trades` and `barter.searches`, and the
    observables `dispersion`, the spread of the logarithms of the rates, and `utility`, the mean utility.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValueError:
            The topology is unknown.

    Examples:
        >>> from skcomplex.systems.economics.markets import WilhiteTradeNetworksSystem
        >>> system = WilhiteTradeNetworksSystem(40, topology='isolated', n_groups=4).prepare().system_
        >>> system.elements[0].prepare().size_
        40
    """

    field = 'economics/markets'
    topics = ('barter', 'trade', 'network', 'small world', 'agents')
    references = ('Wilhite 2001', 'Gooding 2019')
    until = 60
    observables = (
        ObservableSpec('dispersion', group='traders', variable='mrs', reduce=_log_spread),
        ObservableSpec('utility', _get_utility, group='traders'),
    )

    def __init__(
        self,
        n_agents: int = 500,
        *,
        topology: str = 'small_world',
        n_groups: int = 10,
        shortcuts: int = 5,
        endowment: float = 100.0,
        units: int = 100,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_agents = n_agents
        self.topology = topology
        self.n_groups = n_groups
        self.shortcuts = shortcuts
        self.endowment = endowment
        self.units = units
        self.random_state = random_state

    @staticmethod
    def _trade_network(
        topology: str,
        n_agents: int,
        n_groups: int,
        shortcuts: int,
        rng: np.random.Generator,
    ) -> nx.Graph:
        """Return the trade network of one of Wilhite's four topologies.

        Its nodes are numbered from zero.
        """
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

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        graph = self._trade_network(self.topology, self.n_agents, self.n_groups, self.shortcuts, rng)
        goods = rng.integers(1, int(self.endowment) + 1, size=(2, self.n_agents)).astype(float)
        traders = GroupElement(
            'traders',
            element=BarterTraderElement,
            size=self.n_agents,
            values={
                'good1': goods[0],
                'good2': goods[1],
                'mrs': goods[1] / goods[0],
                'links': np.array([graph.degree(node) for node in range(self.n_agents)], dtype=float),
            },
            location=list(range(self.n_agents)),
        )
        return CompositeSystem(
            self.label,
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
                    rule=PartnerMatchingSpec(choice='score', score=BarterTraderElement.get_rate_gap, value='provider'),
                    quantity=('quantity', 'partners'),
                    units=self.units,
                ),
            ],
            space=NetworkSpace(graph),
            time=DiscreteTime(),
        )
