"""Epidemiology models on networks: quenched mean field, adaptive rewiring, and activity-driven temporal networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import networkx as nx
import numpy as np

from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.components import Element, Group
from ....core.interactions import Interaction
from ....core.laws import Jump, Law, Rate, Symbols, Update, add_edge, bernoulli, neighbors, remove_edge, rewire, where
from ....core.runtime import register_model
from ....core.space import DiscreteTime, NetworkSpace
from ....core.system import System

HIDDEN = VisualSpec('sites', {'states': []})


class Carrier(Element):
    """Node of a network whose state is the probability that it is infected, in quenched mean field.

    The node recovers at rate 1 and every infected neighbour infects it at rate `lam`, with the probabilities of the
    neighbours taken as independent, so `dp_i / dt = -p_i + lam (1 - p_i) sum_j A_ij p_j` over the neighbours `j`.

    Args:
        label:
            The label of the node or of the group of nodes.

        p:
            The initial probability of being infected.

        lam:
            The spreading rate, the ratio of the infection rate per edge and the recovery rate.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import Carrier
        >>> Carrier('node').lam
        0.2
    """

    p: float = state(default=0.05, domain=Domain.NONNEGATIVE)
    lam: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the recovery and the infection by the neighbours.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.p, -s.p + s.lam * (1 - s.p) * neighbors(s.other.p))]


class Host(Element):
    """Node of a network that is susceptible at `i = 0` or infected at `i = 1`, in continuous time.

    An infected node recovers at rate `gamma`, and a susceptible node is infected at rate `beta` by every infected
    neighbour.

    Args:
        label:
            The label of the node or of the group of nodes.

        i:
            Whether the node is infected.

        beta:
            The infection rate per edge to an infected neighbour.

        gamma:
            The recovery rate.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import Host
        >>> Host('node').gamma
        1.0
    """

    i: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    beta: float = parameter(default=0.4, domain=Domain.NONNEGATIVE)
    gamma: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the infection by the infected neighbours and the recovery.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            Jump(s.beta * (1 - s.i) * neighbors(s.other.i), {s.i: 1}),
            Jump(s.gamma * s.i, {s.i: -1}),
        ]


class Contact(Element):
    """Edge of a network between two hosts, which carries no state of its own.

    Args:
        label:
            The label of the edge or of the group of edges.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import Contact
        >>> Contact('edge').label
        'edge'
    """


class Rewiring(Interaction):
    """Susceptible end of an edge to an infected node that moves the edge to another susceptible node at rate `w`.

    Args:
        label:
            The label of the interaction.

        hosts:
            The path of the group of hosts on the nodes.

        contacts:
            The path of the variable group on the edges.

        w:
            The rewiring rate of an edge between a susceptible and an infected node.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import Rewiring
        >>> Rewiring('rewiring', hosts='hosts', contacts='contacts', w=2.0).w
        2.0
    """

    over: ClassVar[str] = 'neighbours'
    hosts: str = role()
    contacts: str = role()
    w: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rewiring of the edge from either orientation of a susceptible and an infected end.

        Args:
            s:
                The namespace of the edge, with `s.source`, `s.target`, and `s.edge`.

        Returns:
            laws:
                The jumps that rewire the edge.
        """

        def susceptible(node: Symbols) -> object:
            """Return the weight of a candidate new end, 1 for a susceptible node and 0 otherwise."""
            return 1 - node.i

        return [
            Jump(s.w * (1 - s.source.i) * s.target.i, {rewire(keep='source', toward=susceptible): 1}),
            Jump(s.w * s.source.i * (1 - s.target.i), {rewire(keep='target', toward=susceptible): 1}),
        ]


class ActiveHost(Element):
    """Node of an activity-driven network, susceptible at `i = 0` or infected at `i = 1`, in discrete time.

    At every step the node fires with probability `a`, its activity, and joins `m` other nodes by new edges. An
    infected node recovers with probability `mu`, and a susceptible node is infected with probability `lam` by every
    infected neighbour along the edges of the previous step.

    Args:
        label:
            The label of the node or of the group of nodes.

        i:
            Whether the node is infected.

        a:
            The activity, the probability of firing at a step.

        m:
            The number of edges an active node creates.

        lam:
            The probability of infection along one edge.

        mu:
            The probability of recovery at a step.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import ActiveHost
        >>> ActiveHost('node').m
        2
    """

    i: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    a: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    m: int = 2
    lam: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    mu: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the infection along the edges, the recovery, and the edges created when the node fires.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        escape = (1 - s.lam) ** neighbors(s.other.i)
        infected = bernoulli('infection', 1 - escape)
        recovered = bernoulli('recovery', s.mu)
        return [
            Update(s.i, where(s.i > 1 / 2, 1 - recovered, infected)),
            Update(add_edge(edges=self.m), s.a),
        ]


class Expiry(Interaction):
    """Removal of every edge of an activity-driven network one step after it is created.

    Args:
        label:
            The label of the interaction.

        hosts:
            The path of the group of hosts on the nodes.

        contacts:
            The path of the variable group on the edges.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import Expiry
        >>> Expiry('expiry', hosts='hosts', contacts='contacts').hosts
        'hosts'
    """

    over: ClassVar[str] = 'neighbours'
    hosts: str = role()
    contacts: str = role()

    def laws(self, s: Symbols) -> list[Law]:
        """Return the removal of the edge at the next step.

        Args:
            s:
                The namespace of the edge, with `s.source`, `s.target`, and `s.edge`.

        Returns:
            laws:
                The update that removes the edge.
        """
        return [Update(remove_edge(), 1)]


def _initial_stream(seed: int | None) -> np.random.Generator:
    """Return a random stream of initial values that differs from the run's stream of the same seed."""
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])


def _random_network(n: int, k: float, seed: int | None) -> NetworkSpace:
    """Return an Erdős-Rényi network of `n` nodes with mean degree `k`."""
    return NetworkSpace.erdos_renyi(n, k / (n - 1), seed=seed)


@register_model(
    field='biology/epidemiology',
    tags=('network', 'quenched mean field', 'ode', 'epidemic threshold', 'spectral radius'),
    references=('Pastor-Satorras et al. 2015', 'Wang et al. 2003'),
    defaults={'until': 40.0, 'dt': 0.2},
)
def quenched_mean_field_sis(
    graph: nx.Graph | None = None,
    *,
    n: int = 300,
    m: int = 3,
    lam: float = 0.2,
    p: float = 0.05,
    seed: int | None = 1,
) -> System:
    """Return the SIS epidemic on a fixed network in quenched mean field, one infection probability per node.

    The infection persists above the threshold `lam_c = 1 / Lambda`, the inverse of the largest eigenvalue of the
    adjacency matrix, and dies out below it. On scale-free networks the hubs keep it alive at small `lam`.

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

    Returns:
        system:
            The model, with the probabilities `nodes.p` of the group `nodes` on the nodes of the network.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.biology.epidemiology import quenched_mean_field_sis
        >>> quenched_mean_field_sis(nx.cycle_graph(10)).elements[0].size
        10
    """
    space = NetworkSpace(graph) if graph is not None else NetworkSpace.barabasi_albert(n, m, seed=seed)
    return System(components=[Carrier.lattice('nodes', space, p=p, lam=lam)])


@register_model(
    field='biology/epidemiology',
    tags=('network', 'adaptive network', 'rewiring', 'jumps', 'epidemic threshold'),
    references=("Gross, Dommar D'Lima, and Blasius 2006", 'Thurner et al. 2018', 'Pastor-Satorras et al. 2015'),
    defaults={'until': 30.0, 'dt': 0.25},
)
def adaptive_sis(
    n: int = 200,
    k: float = 6.0,
    *,
    beta: float = 0.4,
    gamma: float = 1.0,
    w: float = 1.0,
    infected: float = 0.2,
    seed: int | None = 1,
) -> System:
    """Return the SIS epidemic on a network whose susceptible nodes cut their edges to infected neighbours.

    Every edge between a susceptible and an infected node is rewired at rate `w`: the susceptible end keeps it and
    joins it to another susceptible node. Rewiring isolates the infected nodes and raises the epidemic threshold above
    its value on the fixed network, while the susceptible nodes cluster.

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

    Returns:
        system:
            The model, with the states `hosts.i` of the variable group `hosts` on the nodes and the variable group
            `contacts` on the edges.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import adaptive_sis
        >>> [component.label for component in adaptive_sis(n=20).elements]
        ['hosts', 'contacts']
    """
    space = _random_network(n, k, seed)
    initial = (_initial_stream(seed).random(n) < infected).astype(float)
    hosts = Group(
        'hosts',
        element=Host,
        space=space,
        on='sites',
        variable=True,
        capacity=n,
        values={'i': initial, 'beta': beta, 'gamma': gamma},
        visual=VisualSpec('sites', {'states': ['i'], 'low': 0, 'high': 1}),
    )
    edges = space.graph.number_of_edges()
    contacts = Group(
        'contacts',
        element=Contact,
        space=space,
        on='edges',
        variable=True,
        capacity=edges,
        visual=HIDDEN,
    )
    rewiring = Rewiring('rewiring', hosts='hosts', contacts='contacts', w=w)
    return System(components=[hosts, contacts], interactions=[rewiring])


@register_model(
    field='biology/epidemiology',
    tags=('temporal network', 'activity-driven', 'stochastic map', 'epidemic threshold'),
    references=('Perra et al. 2012', 'Pastor-Satorras et al. 2015'),
    defaults={'until': 150, 'dt': 1.0},
)
def activity_driven_epidemics(
    n: int = 500,
    *,
    m: int = 2,
    exponent: float = 2.2,
    least: float = 0.01,
    lam: float = 0.9,
    mu: float = 0.1,
    infected: float = 0.05,
    seed: int | None = 1,
) -> System:
    """Return SIS spreading on an activity-driven temporal network, whose edges last one step.

    At every step every node fires with probability its activity, drawn from `a^-exponent` above `least`, and joins
    `m` random nodes. The infection travels along the edges of a step, which disappear at the next step, so it spreads
    only when `lam / mu` exceeds a threshold set by the moments of the activity, `2 <a> / (m (<a> + sqrt(<a^2>)))` in
    the limit of small probabilities, and not by the degrees of the network aggregated over time.

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

    Returns:
        system:
            The model, with the states `hosts.i` and activities `hosts.a` of the group `hosts` on the nodes, and the
            variable group `contacts` on the edges of every step.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import activity_driven_epidemics
        >>> float(activity_driven_epidemics(n=10, seed=0).elements[0].values['a'].min()) >= 0.01
        True
    """
    rng = _initial_stream(seed)
    activities = np.minimum(least * (1 - rng.random(n)) ** (-1 / (exponent - 1)), 1.0)
    initial = (rng.random(n) < infected).astype(float)
    space = NetworkSpace(nx.empty_graph(n))
    hosts = Group(
        'hosts',
        element=ActiveHost,
        space=space,
        on='sites',
        variable=True,
        capacity=n,
        values={'i': initial, 'a': activities, 'm': m, 'lam': lam, 'mu': mu},
        visual=VisualSpec('sites', {'states': ['i'], 'low': 0, 'high': 1}),
    )
    contacts = Group(
        'contacts',
        element=Contact,
        space=space,
        on='edges',
        variable=True,
        capacity=2 * m * n,
        visual=HIDDEN,
    )
    return System(
        components=[hosts, contacts],
        interactions=[Expiry('expiry', hosts='hosts', contacts='contacts')],
        time=DiscreteTime(),
    )
