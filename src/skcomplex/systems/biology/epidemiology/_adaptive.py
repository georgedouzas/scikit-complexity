"""Epidemiology systems on networks.

They cover quenched mean field, adaptive rewiring, and activity-driven temporal networks.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, JumpLaw, TopologyRequestSpec, UpdateLaw
from ....core.components.laws.utils import neighbors, uniform, where
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Array
from ...general.dynamics import NetworkDynamicalSystem
from ...general.dynamics.types import Equations
from ..common.networks import HostElement, SpreadingInteraction


class AdaptiveHostElement(HostElement):
    """Host of an adaptive network, drawn by whether it is infected."""

    visual = VisualSpec('sites', {'states': ['i'], 'low': 0, 'high': 1})


class ContactEdgeElement(Element):
    """Contact between two hosts, hidden from the drawing."""

    visual = VisualSpec('sites', {'states': []})


class ActiveHostElement(Element):
    """Node of an activity-driven network, susceptible or infected, in discrete time.

    The node is susceptible at `i = 0` and infected at `i = 1`.

    One interaction recovers an infected node with probability `mu` in a step, and another infects a susceptible node
    with probability `lam` by every infected neighbour along the edges of the previous step. A third joins the node to
    other nodes with the probability `a`, its activity.

    Args:
        label:
            The label of the node or of the group of nodes.

        i:
            Whether the node is infected.

        a:
            The activity, the probability of firing at a step.

        lam:
            The probability of infection along one edge.

        mu:
            The probability of recovery at a step.

        space:
            The network whose nodes hold one member each, or `None` for one node.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import ActiveHostElement
        >>> ActiveHostElement('node').mu
        0.1
    """

    visual = VisualSpec('sites', {'states': ['i'], 'low': 0, 'high': 1})

    def __init__(
        self,
        label: str,
        *,
        i: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.0,
        a: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.1,
        lam: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.3,
        mu: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.1,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.i = i
        self.a = a
        self.lam = lam
        self.mu = mu


class QuenchedMeanFieldSISSystem(NetworkDynamicalSystem):
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
        >>> QuenchedMeanFieldSISSystem(nx.cycle_graph(10)).prepare().simulate(until=0.0).get_values('nodes.p').shape
        (1, 10)
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.graph = graph
        self.n = n
        self.m = m
        self.lam = lam
        self.p = p
        self.seed = seed

    def _get_graph(self) -> nx.Graph:
        """Return the graph, a Barabási-Albert network of `n` nodes by default."""
        return self.graph if self.graph is not None else nx.barabasi_albert_graph(self.n, self.m, seed=self.seed)

    def _get_element_label(self) -> str:
        """Return the label of the field of the nodes."""
        return 'nodes'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        return {
            'p': self.p,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the value of every parameter by name."""
        return {
            'lam': self.lam,
        }

    def _get_equations(self) -> Equations:
        """Return the recovery of every node and its infection by its neighbours."""
        return {'p': lambda s: -s.target.p + s.target.lam * (1 - s.target.p) * neighbors(s.target.other.p)}


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
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

    def _get_rewiring(self, keep: str) -> Interaction:
        """Return the rewiring of edges from a susceptible end `keep` to an infected end."""
        other = 'target' if keep == 'source' else 'source'

        def rewire(edge: SimpleNamespace) -> list[BaseLaw]:
            """Return the rewiring of the edge at the rate of its susceptible end.

            The susceptible end rewires the edge away from the infected end.
            """
            kept, left = getattr(edge, keep), getattr(edge, other)
            request = TopologyRequestSpec('rewire', toward=self._get_susceptible, keep=keep)
            return [JumpLaw(kept.w * (1 - kept.i) * left.i, {request: 1})]

        return Interaction(f'rewiring_{keep}', roles={'nodes': 'hosts'}, laws=rewire, along=True)

    @staticmethod
    def _get_random_network(n: int, k: float, seed: int | None) -> NetworkSpace:
        """Return an Erdős-Rényi network of `n` nodes with mean degree `k`."""
        return NetworkSpace(nx.gnp_random_graph(n, k / (n - 1), seed=seed))

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the SIS epidemic on a network.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        space = self._get_random_network(self.n, self.k, self.seed)
        initial = (np.random.default_rng(self.seed).random(self.n) < self.infected).astype(float)
        hosts = AdaptiveHostElement('hosts', i=initial, w=self.w, space=space)
        contacts = ContactEdgeElement('contacts', space=space, on='edges')
        spreading = SpreadingInteraction('spreading', target='hosts', state='i', beta=self.beta, gamma=self.gamma)
        self.system_ = System(
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import ActivityDrivenEpidemicsSystem
        >>> float(ActivityDrivenEpidemicsSystem(n=10, seed=0).prepare().system_.elements[0].a.min()) >= 0.01
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n = n
        self.m = m
        self.exponent = exponent
        self.least = least
        self.lam = lam
        self.mu = mu
        self.infected = infected
        self.seed = seed

    def _get_firing(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the new edges of a node that fires at a step with the probability of its activity."""
        return [UpdateLaw(TopologyRequestSpec('add-edge', self.m), s.source.a)]

    @staticmethod
    def _get_expiry(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the removal of every edge at the next step."""
        return [UpdateLaw(TopologyRequestSpec('remove-edge'), 1)]

    @staticmethod
    def _get_infection(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the infection of a node by at least one infected neighbour along its edges."""
        i = s.target.i
        probability = 1 - (1 - s.target.lam) ** neighbors(s.target.other.i)
        moves = uniform('transition_infection') < probability
        return [UpdateLaw(i, where(i == 0, where(moves, 1, 0), i), source=0)]

    @staticmethod
    def _get_recovery(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the recovery of an infected node with probability `mu` in a step."""
        i = s.target.i
        moves = uniform('transition_recovery') < s.target.mu
        return [UpdateLaw(i, where(i == 1, where(moves, 0, 1), i), source=1)]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of SIS spreading on an activity-driven temporal network.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.seed)
        activities = np.minimum(self.least * (1 - rng.random(self.n)) ** (-1 / (self.exponent - 1)), 1.0)
        initial = (rng.random(self.n) < self.infected).astype(float)
        space = NetworkSpace(nx.empty_graph(self.n))
        hosts = ActiveHostElement('hosts', i=initial, a=activities, lam=self.lam, mu=self.mu, space=space)
        contacts = ContactEdgeElement('contacts', space=space, on='edges')
        self.system_ = System(
            self.label,
            elements=[hosts, contacts],
            interactions=[
                Interaction('firing', roles={'nodes': 'hosts'}, laws=self._get_firing, along=True),
                Interaction('expiry', roles={'nodes': 'hosts'}, laws=self._get_expiry, along=True),
                Interaction('infection', roles={'target': 'hosts'}, laws=self._get_infection),
                Interaction('recovery', roles={'target': 'hosts'}, laws=self._get_recovery),
            ],
        )
