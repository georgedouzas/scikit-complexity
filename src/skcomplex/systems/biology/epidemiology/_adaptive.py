"""Epidemiology systems on networks: quenched mean field, adaptive rewiring, and activity-driven temporal networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import Element, GroupElement, LatticeElement
from ....core.components.interactions import TopologyInteraction, TransitionInteraction
from ....core.components.laws.utils import neighbors
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import System
from ....core.specs import VisualSpec
from ..._catalog import register_system
from ._active_host import ActiveHostElement
from ._carrier import CarrierElement
from ._host import HostElement
from ._spreading import SpreadingInteraction

HIDDEN = VisualSpec('sites', {'states': []})


def _get_susceptible(node: SimpleNamespace) -> object:
    """Return the weight of a candidate new end of an edge, 1 for a susceptible node and 0 otherwise."""
    return 1 - node.i


def _get_activity(node: SimpleNamespace) -> object:
    """Return the probability that a node fires at a step, its activity."""
    return node.source.a


def _get_rewiring(keep: str) -> TopologyInteraction:
    """Return the rewiring of the edges whose end `keep` is susceptible and whose other end is infected."""
    other = 'target' if keep == 'source' else 'source'

    def rate(edge: SimpleNamespace) -> object:
        """Return the rate at which the susceptible end rewires the edge away from the infected end."""
        kept, left = getattr(edge, keep), getattr(edge, other)
        return kept.w * (1 - kept.i) * left.i

    return TopologyInteraction(
        f'rewiring_{keep}',
        target='hosts',
        action='rewire',
        rate=rate,
        toward=_get_susceptible,
        keep=keep,
    )


def _get_infection(s: SimpleNamespace) -> Expression:
    """Return the probability that at least one infected neighbour infects a node along its edges."""
    probability: Expression = 1 - (1 - s.target.lam) ** neighbors(s.target.other.i)
    return probability


def _get_recovery(s: SimpleNamespace) -> Expression:
    """Return the probability that an infected node recovers in a step."""
    probability: Expression = s.target.mu
    return probability


def _random_network(n: int, k: float, seed: int | None) -> NetworkSpace:
    """Return an Erdős-Rényi network of `n` nodes with mean degree `k`."""
    return NetworkSpace(nx.gnp_random_graph(n, k / (n - 1), seed=seed))


@register_system(
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
            The system, with the probabilities `nodes.p` of the group `nodes` on the nodes of the network.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.biology.epidemiology import quenched_mean_field_sis
        >>> quenched_mean_field_sis(nx.cycle_graph(10)).elements[0].prepare().size_
        10
    """
    space = NetworkSpace(graph) if graph is not None else NetworkSpace(nx.barabasi_albert_graph(n, m, seed=seed))
    return System(
        elements=[LatticeElement('nodes', element=CarrierElement, space=space, values={'p': p, 'lam': lam})],
    )


@register_system(
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
            The system, with the states `hosts.i` of the variable group `hosts` on the nodes and the variable group
            `contacts` on the edges.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import adaptive_sis
        >>> [component.label for component in adaptive_sis(n=20).elements]
        ['hosts', 'contacts']
    """
    space = _random_network(n, k, seed)
    initial = (np.random.default_rng(seed).random(n) < infected).astype(float)
    hosts = GroupElement(
        'hosts',
        element=HostElement,
        space=space,
        on='sites',
        variable=True,
        capacity=n,
        values={'i': initial, 'w': w},
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
    spreading = SpreadingInteraction('spreading', target='hosts', state='i', beta=beta, gamma=gamma)
    return System(
        elements=[hosts, contacts],
        interactions=[spreading, _get_rewiring('source'), _get_rewiring('target')],
    )


@register_system(
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
            The system, with the states `hosts.i` and activities `hosts.a` of the group `hosts` on the nodes, and the
            variable group `contacts` on the edges of every step.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import activity_driven_epidemics
        >>> float(activity_driven_epidemics(n=10, seed=0).elements[0].values['a'].min()) >= 0.01
        True
    """
    rng = np.random.default_rng(seed)
    activities = np.minimum(least * (1 - rng.random(n)) ** (-1 / (exponent - 1)), 1.0)
    initial = (rng.random(n) < infected).astype(float)
    space = NetworkSpace(nx.empty_graph(n))
    hosts = GroupElement(
        'hosts',
        element=ActiveHostElement,
        space=space,
        on='sites',
        variable=True,
        capacity=n,
        values={'i': initial, 'a': activities, 'lam': lam, 'mu': mu},
        visual=VisualSpec('sites', {'states': ['i'], 'low': 0, 'high': 1}),
    )
    contacts = GroupElement(
        'contacts',
        element=Element,
        space=space,
        on='edges',
        variable=True,
        capacity=2 * m * n,
        visual=HIDDEN,
    )
    return System(
        elements=[hosts, contacts],
        interactions=[
            TopologyInteraction('firing', target='hosts', action='add-edge', probability=_get_activity, edges=m),
            TopologyInteraction('expiry', target='hosts', action='remove-edge'),
            TransitionInteraction(
                'infection',
                target='hosts',
                state='i',
                source=0,
                destination=1,
                probability=_get_infection,
            ),
            TransitionInteraction(
                'recovery',
                target='hosts',
                state='i',
                source=1,
                destination=0,
                probability=_get_recovery,
            ),
        ],
        time=DiscreteTime(),
    )
