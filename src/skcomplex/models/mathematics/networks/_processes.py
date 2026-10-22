"""Network models: random walks, diffusion, and PageRank on graphs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import networkx as nx
import numpy as np

from ....core.base import Domain, VisualSpec, parameter, state
from ....core.model import (
    AgentType,
    Element,
    Field,
    Law,
    NetworkSpace,
    Rate,
    Rule,
    Symbols,
    System,
    Update,
    laplacian,
)
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

NodeContent = Element.define(
    'NodeContent',
    states={'u': 0.0},
    parameters={'D': parameter(1.0, domain=Domain.NONNEGATIVE)},
    laws=lambda s: [Rate(s.u, s.D * laplacian(s.u))],
)

NodeRank = Element.define(
    'NodeRank',
    states={'rank': state(default=0.0, domain=Domain.NONNEGATIVE)},
    parameters={
        'damping': parameter(0.85, domain=Domain.NONNEGATIVE),
        'degree': parameter(1.0, domain=Domain.POSITIVE),
        'n_nodes': parameter(1.0, domain=Domain.POSITIVE),
    },
    laws=lambda s: [
        Update(s.rank, (1 - s.damping) / s.n_nodes + s.damping * (laplacian(s.rank / s.degree) + s.rank)),
    ],
)


class NetworkWalker(AgentType):
    """Walker that moves to a uniformly random neighbouring node of a network at every step.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial node, or `None` for a random node.

        steps:
            The initial number of steps taken.

    Examples:
        >>> from skcomplex.models.mathematics.networks import NetworkWalker
        >>> NetworkWalker.group('walkers', 5)
        Group('walkers', element=NetworkWalker, size=5)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'steps'})
    steps: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the stepping rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.step)]

    def step(self, agent: AgentView, model: ModelView) -> None:
        """Move to a random neighbouring node.

        Args:
            agent:
                The walker.

            model:
                The model.
        """
        cells = agent.neighbor_cells()
        if not cells:
            return
        agent.move_to(cells[int(model.rng.integers(len(cells)))])
        agent.steps += 1


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
        >>> network_random_walk(nx.cycle_graph(6), n_walkers=3).elements[0].size
        3
    """
    graph = _graph_or_default(graph)
    return System(components=[NetworkWalker.group('walkers', n_walkers)], space=NetworkSpace(graph))


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
        >>> network_diffusion(nx.path_graph(4)).elements[0].size
        4
    """
    graph = _graph_or_default(graph)
    initial = np.zeros(graph.number_of_nodes())
    initial[source] = 1.0
    nodes = Field('nodes', NodeContent, NetworkSpace(graph), values={'u': initial, 'D': D})
    return System(components=[nodes])


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
        >>> pagerank(nx.cycle_graph(5)).elements[0].size
        5
    """
    graph = _graph_or_default(graph)
    size = graph.number_of_nodes()
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    values = {'rank': np.full(size, 1 / size), 'degree': degrees, 'n_nodes': float(size), 'damping': damping}
    return System(components=[Field('pages', NodeRank, NetworkSpace(graph), values=values)])
