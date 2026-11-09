"""Epidemiology systems on networks.

They cover quenched mean field, adaptive rewiring, and activity-driven temporal networks.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import TopologyInteraction, TransitionInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import neighbors
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.systems import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..common.networks import HostElement, SpreadingInteraction

HIDDEN = VisualSpec('sites', {'states': []})


class ActiveHostElement(Element):
    """Node of an activity-driven network, susceptible or infected, in discrete time.

    The node is susceptible at `i = 0` and infected at `i = 1`.

    A `TransitionInteraction` recovers an infected node with probability `mu` in a step, and another infects a
    susceptible node with probability `lam` by every infected neighbour along the edges of the previous step. A
    `TopologyInteraction` joins the node to other nodes with the probability `a`, its activity.

    Args:
        label:
            The label of the node or of the group of nodes.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        i:
            Whether the node is infected.

        a:
            The activity, the probability of firing at a step.

        lam:
            The probability of infection along one edge.

        mu:
            The probability of recovery at a step.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import ActiveHostElement
        >>> ActiveHostElement('node').mu
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        i: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        lam: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        mu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.i = i
        self.a = a
        self.lam = lam
        self.mu = mu


class CarrierElement(Element):
    """Node of a network holding its infection probability, in quenched mean field.

    The node recovers at rate 1 and every infected neighbour infects it at rate `lam`, with the probabilities of the
    neighbours taken as independent, so `dp_i / dt = -p_i + lam (1 - p_i) sum_j A_ij p_j` over the neighbours `j`.

    Args:
        label:
            The label of the node or of the group of nodes.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        p:
            The initial probability of being infected.

        lam:
            The spreading rate, the ratio of the infection rate per edge and the recovery rate.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import CarrierElement
        >>> CarrierElement('node').lam
        0.2
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        lam: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.p = p
        self.lam = lam

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the recovery and the infection by the neighbours.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.p, -s.p + s.lam * (1 - s.p) * neighbors(s.other.p))]


class QuenchedMeanFieldSISSystem(BaseSystem):
    """The SIS epidemic on a fixed network in quenched mean field.

    It has one infection probability per node.

    The infection persists above the threshold `lam_c = 1 / Lambda`, the inverse of the largest eigenvalue of the
    adjacency matrix, and dies out below it. On scale-free networks the hubs keep it alive at small `lam`. The system
    has the probabilities `nodes.p` of the group `nodes` on the nodes of the network.

    Args:
        graph:
            The network, or `None` for a Barabási-Albert network of `n` nodes with `m` edges per new node.

        n:
            The number of nodes of the default network.

        m:
            The number of edges of every new node of the default network.

        lam:
            The spreading rate, the ratio of the infection rate per edge and the recovery rate.

        p:
            The initial probability that every node is infected.

        seed:
            The seed of the default network.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.biology.epidemiology import QuenchedMeanFieldSISSystem
        >>> QuenchedMeanFieldSISSystem(nx.cycle_graph(10)).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'biology/epidemiology'
    topics = ('network', 'quenched mean field', 'ode', 'epidemic threshold', 'spectral radius')
    references = ('Pastor-Satorras et al. 2015', 'Wang et al. 2003')
    until = 40.0
    dt = 0.2

    def __init__(
        self,
        graph: nx.Graph | None = None,
        *,
        n: int = 300,
        m: int = 3,
        lam: float = 0.2,
        p: float = 0.05,
        seed: int | None = 1,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.graph = graph
        self.n = n
        self.m = m
        self.lam = lam
        self.p = p
        self.seed = seed

    def _build_system(self) -> System:
        """Return the parts of the SIS epidemic on a fixed network in quenched mean field."""
        space = (
            NetworkSpace(self.graph)
            if self.graph is not None
            else NetworkSpace(nx.barabasi_albert_graph(self.n, self.m, seed=self.seed))
        )
        return System(
            self.label,
            elements=[
                GroupElement(
                    'nodes',
                    element=CarrierElement,
                    space=space,
                    values={'p': self.p, 'lam': self.lam},
                    on='sites',
                ),
            ],
        )


class AdaptiveSISSystem(BaseSystem):
    """The SIS epidemic where susceptible nodes cut their edges to infected neighbours.

    Every edge between a susceptible and an infected node is rewired at rate `w`: the susceptible end keeps it and joins
    it to another susceptible node. Rewiring isolates the infected nodes and raises the epidemic threshold above its
    value on the fixed network, while the susceptible nodes cluster. The system has the states `hosts.i` of the variable
    group `hosts` on the nodes and the variable group `contacts` on the edges.

    Args:
        n:
            The number of nodes.

        k:
            The mean degree of the initial Erdős-Rényi network.

        beta:
            The infection rate per edge to an infected neighbour.

        gamma:
            The recovery rate.

        w:
            The rewiring rate of an edge between a susceptible and an infected node.

        infected:
            The initial share of infected nodes.

        seed:
            The seed of the network and of the initial infections.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import AdaptiveSISSystem
        >>> [component.label for component in AdaptiveSISSystem(n=20).prepare().system_.elements]
        ['hosts', 'contacts']
    """

    field = 'biology/epidemiology'
    topics = ('network', 'adaptive network', 'rewiring', 'jumps', 'epidemic threshold')
    references = ("Gross, Dommar D'Lima, and Blasius 2006", 'Thurner et al. 2018', 'Pastor-Satorras et al. 2015')
    until = 30.0
    dt = 0.25

    def __init__(
        self,
        n: int = 200,
        k: float = 6.0,
        *,
        beta: float = 0.4,
        gamma: float = 1.0,
        w: float = 1.0,
        infected: float = 0.2,
        seed: int | None = 1,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.k = k
        self.beta = beta
        self.gamma = gamma
        self.w = w
        self.infected = infected
        self.seed = seed

    @staticmethod
    def _get_susceptible(node: SimpleNamespace) -> object:
        """Return the weight of a candidate new end of an edge.

        The weight is 1 for a susceptible node and 0 otherwise.
        """
        return 1 - node.i

    def _get_rewiring(self, keep: str) -> TopologyInteraction:
        """Return the rewiring of edges from a susceptible end `keep` to an infected end."""
        other = 'target' if keep == 'source' else 'source'

        def rate(edge: SimpleNamespace) -> object:
            """Return the rate at which the susceptible end rewires the edge.

            The susceptible end rewires the edge away from the infected end.
            """
            kept, left = getattr(edge, keep), getattr(edge, other)
            return kept.w * (1 - kept.i) * left.i

        return TopologyInteraction(
            f'rewiring_{keep}',
            target='hosts',
            action='rewire',
            rate=rate,
            toward=self._get_susceptible,
            keep=keep,
        )

    @staticmethod
    def _random_network(n: int, k: float, seed: int | None) -> NetworkSpace:
        """Return an Erdős-Rényi network of `n` nodes with mean degree `k`."""
        return NetworkSpace(nx.gnp_random_graph(n, k / (n - 1), seed=seed))

    def _build_system(self) -> System:
        """Return the parts of the SIS epidemic on a network."""
        space = self._random_network(self.n, self.k, self.seed)
        initial = (np.random.default_rng(self.seed).random(self.n) < self.infected).astype(float)
        hosts = GroupElement(
            'hosts',
            element=HostElement,
            space=space,
            on='sites',
            variable=True,
            capacity=self.n,
            values={'i': initial, 'w': self.w},
            visual=VisualSpec('sites', {'states': ['i'], 'low': 0, 'high': 1}),
        )
        edges = space.graph.number_of_edges()
        contacts = GroupElement(
            'contacts',
            element=Element,
            space=space,
            on='edges',
            variable=True,
            capacity=edges,
            visual=HIDDEN,
        )
        spreading = SpreadingInteraction('spreading', target='hosts', state='i', beta=self.beta, gamma=self.gamma)
        return System(
            self.label,
            elements=[hosts, contacts],
            interactions=[spreading, self._get_rewiring('source'), self._get_rewiring('target')],
        )


class ActivityDrivenEpidemicsSystem(BaseSystem):
    """SIS spreading on an activity-driven temporal network, whose edges last one step.

    At every step every node fires with probability its activity, drawn from `a^-exponent` above `least`, and joins `m`
    random nodes. The infection travels along the edges of a step, which disappear at the next step, so it spreads only
    when `lam / mu` exceeds a threshold set by the moments of the activity, `2 <a> / (m (<a> + sqrt(<a^2>)))` in the
    limit of small probabilities, and not by the degrees of the network aggregated over time. The system has the states
    `hosts.i` and activities `hosts.a` of the group `hosts` on the nodes, and the variable group `contacts` on the edges
    of every step.

    Args:
        n:
            The number of nodes.

        m:
            The number of edges an active node creates.

        exponent:
            The exponent of the power-law distribution of activities.

        least:
            The smallest activity.

        lam:
            The probability of infection along one edge.

        mu:
            The probability of recovery at a step.

        infected:
            The initial share of infected nodes.

        seed:
            The seed of the activities and the initial infections.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import ActivityDrivenEpidemicsSystem
        >>> float(ActivityDrivenEpidemicsSystem(n=10, seed=0).prepare().system_.elements[0].values['a'].min()) >= 0.01
        True
    """

    field = 'biology/epidemiology'
    topics = ('temporal network', 'activity-driven', 'stochastic map', 'epidemic threshold')
    references = ('Perra et al. 2012', 'Pastor-Satorras et al. 2015')
    until = 150
    dt = 1.0

    def __init__(
        self,
        n: int = 500,
        *,
        m: int = 2,
        exponent: float = 2.2,
        least: float = 0.01,
        lam: float = 0.9,
        mu: float = 0.1,
        infected: float = 0.05,
        seed: int | None = 1,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.m = m
        self.exponent = exponent
        self.least = least
        self.lam = lam
        self.mu = mu
        self.infected = infected
        self.seed = seed

    @staticmethod
    def _get_activity(node: SimpleNamespace) -> object:
        """Return the probability that a node fires at a step, its activity."""
        return node.source.a

    @staticmethod
    def _get_infection(s: SimpleNamespace) -> Expression:
        """Return the probability that at least one infected neighbour infects a node.

        The neighbours infect the node along its edges.
        """
        probability: Expression = 1 - (1 - s.target.lam) ** neighbors(s.target.other.i)
        return probability

    @staticmethod
    def _get_recovery(s: SimpleNamespace) -> Expression:
        """Return the probability that an infected node recovers in a step."""
        probability: Expression = s.target.mu
        return probability

    def _build_system(self) -> System:
        """Return the parts of SIS spreading on an activity-driven temporal network.

        The edges of the network last one step.
        """
        rng = np.random.default_rng(self.seed)
        activities = np.minimum(self.least * (1 - rng.random(self.n)) ** (-1 / (self.exponent - 1)), 1.0)
        initial = (rng.random(self.n) < self.infected).astype(float)
        space = NetworkSpace(nx.empty_graph(self.n))
        hosts = GroupElement(
            'hosts',
            element=ActiveHostElement,
            space=space,
            on='sites',
            variable=True,
            capacity=self.n,
            values={'i': initial, 'a': activities, 'lam': self.lam, 'mu': self.mu},
            visual=VisualSpec('sites', {'states': ['i'], 'low': 0, 'high': 1}),
        )
        contacts = GroupElement(
            'contacts',
            element=Element,
            space=space,
            on='edges',
            variable=True,
            capacity=2 * self.m * self.n,
            visual=HIDDEN,
        )
        return System(
            self.label,
            elements=[hosts, contacts],
            interactions=[
                TopologyInteraction(
                    'firing',
                    target='hosts',
                    action='add-edge',
                    probability=self._get_activity,
                    edges=self.m,
                ),
                TopologyInteraction('expiry', target='hosts', action='remove-edge'),
                TransitionInteraction(
                    'infection',
                    target='hosts',
                    state='i',
                    source=0,
                    destination=1,
                    probability=self._get_infection,
                ),
                TransitionInteraction(
                    'recovery',
                    target='hosts',
                    state='i',
                    source=1,
                    destination=0,
                    probability=self._get_recovery,
                ),
            ],
            time=DiscreteTime(),
        )
