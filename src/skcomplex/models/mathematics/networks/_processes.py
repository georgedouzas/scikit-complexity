"""Network models: random walks, diffusion, and PageRank on graphs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import networkx as nx
import numpy as np

from ....core.elements import FieldElement, GroupElement
from ....core.interactions import CouplingInteraction, DiffusionInteraction, MovementInteraction
from ....core.spaces import NetworkSpace
from ....core.system import System
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ..._catalog import register_model
from ._network_walker import NetworkWalkerElement
from ._network_walker_view import NetworkWalkerView
from ._node_content import NodeContentElement
from ._node_rank import NodeRankElement


def _count_step(agent: AgentViewProtocol, model: ModelViewProtocol) -> None:
    """Count a step the walker takes."""
    if isinstance(agent, NetworkWalkerView):
        agent.steps += 1


def _get_link_term(s: SimpleNamespace) -> object:
    """Return what one link brings to the rank of a page: a share of the rank of the other page and of the jumps."""
    member, other = s.member, s.other
    return member.damping * other.rank / other.degree + (1 - member.damping) / (member.n_nodes * member.degree)


def _graph_or_default(graph: nx.Graph | None) -> nx.Graph:
    """Return the graph, or Zachary's karate club when none is given."""
    return nx.karate_club_graph() if graph is None else graph


@register_model(
    field='mathematics/networks',
    tags=('network', 'random walk', 'agents'),
    references=('Thurner et al. 2018', 'Jensen 2023'),
    defaults={'until': 100},
)
def network_random_walk(graph: nx.Graph | None = None, *, n_walkers: int = 50) -> System:
    """Return random walkers on a network, which visit every node of a connected graph in proportion to its degree.

    Args:
        graph:
            The undirected NetworkX graph, or `None` for Zachary's karate club.

        n_walkers:
            The number of walkers, started at random nodes.

    Returns:
        system:
            The model, with the group `walkers`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.mathematics.networks import network_random_walk
        >>> network_random_walk(nx.cycle_graph(6), n_walkers=3).elements[0].prepare().size_
        3
    """
    graph = _graph_or_default(graph)
    walk = MovementInteraction('walk', target='walkers', step='neighbor', cost=_count_step)
    return System(
        elements=[GroupElement('walkers', element=NetworkWalkerElement, size=n_walkers)],
        interactions=[walk],
        space=NetworkSpace(graph),
    )


@register_model(
    field='mathematics/networks',
    tags=('network', 'diffusion', 'graph laplacian', 'ode'),
    references=('Thurner et al. 2018', 'Jensen 2023'),
    defaults={'until': 10.0, 'dt': 0.1},
)
def network_diffusion(graph: nx.Graph | None = None, *, D: float = 1.0, source: int = 0) -> System:
    """Return diffusion on a network, `du_i/dt = D Σ_j A_ij (u_j - u_i)`, which spreads a unit mass to every node.

    The total is conserved, and on a connected graph the content of every node tends to `1 / n`.

    Args:
        graph:
            The undirected NetworkX graph, or `None` for Zachary's karate club.

        D:
            The diffusion coefficient of every link.

        source:
            The index, in node order, of the node that holds the whole mass at the start.

    Returns:
        system:
            The model, with the contents `nodes.u`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.mathematics.networks import network_diffusion
        >>> network_diffusion(nx.path_graph(4)).elements[0].prepare().size_
        4
    """
    graph = _graph_or_default(graph)
    initial = np.zeros(graph.number_of_nodes())
    initial[source] = 1.0
    nodes = FieldElement('nodes', element=NodeContentElement, space=NetworkSpace(graph), values={'u': initial, 'D': D})
    diffusion = DiffusionInteraction('diffusion', target='nodes', state='u', coefficient=lambda s: s.target.D)
    return System(elements=[nodes], interactions=[diffusion])


@register_model(
    field='mathematics/networks',
    tags=('network', 'pagerank', 'centrality', 'discrete map'),
    references=('Brin and Page 1998', 'Thurner et al. 2018', 'Jensen 2023'),
    defaults={'until': 50, 'dt': 1},
)
def pagerank(graph: nx.Graph | None = None, *, damping: float = 0.85) -> System:
    """Return the PageRank iteration `r_i' = (1 - d) / n + d Σ_j A_ij r_j / k_j` on an undirected network.

    The ranks start uniform, stay normalized, and converge to the PageRank of the graph. Every node needs a link.

    Args:
        graph:
            The undirected NetworkX graph without isolated nodes, or `None` for Zachary's karate club.

        damping:
            The probability `d` that the surfer follows a link rather than jumping to a random node.

    Returns:
        system:
            The model, with the ranks `pages.rank`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.mathematics.networks import pagerank
        >>> pagerank(nx.cycle_graph(5)).elements[0].prepare().size_
        5
    """
    graph = _graph_or_default(graph)
    size = graph.number_of_nodes()
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    values = {'rank': np.full(size, 1 / size), 'degree': degrees, 'n_nodes': float(size), 'damping': damping}
    pages = FieldElement('pages', element=NodeRankElement, space=NetworkSpace(graph), values=values)
    links = CouplingInteraction('links', target='pages', read='rank', write='rank', function=_get_link_term, form='map')
    return System(elements=[pages], interactions=[links])
