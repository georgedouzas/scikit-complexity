"""Network models: random walks, diffusion, and PageRank on graphs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, FieldElement, GroupElement
from ....core.laws import BaseLaw, RateLaw, RuleLaw, UpdateLaw
from ....core.laws.utils import laplacian
from ....core.spaces import NetworkSpace
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import AgentView, ModelView
from ....specs import ParameterSpec, StateSpec, VisualSpec


class NodeContentElement(Element):
    """Node content element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec()] = 0.0,
        D: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.D = D

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the node content element."""
        return [RateLaw(s.u, s.D * laplacian(s.u))]


class NodeRankElement(Element):
    """Node rank element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rank: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        damping: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.85,
        degree: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        n_nodes: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rank = rank
        self.damping = damping
        self.degree = degree
        self.n_nodes = n_nodes

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the node rank element."""
        return [
            UpdateLaw(s.rank, (1 - s.damping) / s.n_nodes + s.damping * (laplacian(s.rank / s.degree) + s.rank)),
        ]


class NetworkWalkerElement(Element):
    """Walker that moves to a uniformly random neighbouring node of a network at every step.

    Args:
        label:
            The label of the agent or group.

        steps:
            The initial number of steps taken.

    Examples:
        >>> from skcomplex.models.mathematics.networks import NetworkWalkerElement
        >>> group = GroupElement('walkers', element=NetworkWalkerElement, size=5)
        >>> group.label, group.element.__name__, group.size
        ('walkers', 'NetworkWalkerElement', 5)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'steps'}),
        *,
        steps: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.steps = steps

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the stepping rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.step)]

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
        >>> network_random_walk(nx.cycle_graph(6), n_walkers=3).elements[0].check().prepare().size_
        3
    """
    graph = _graph_or_default(graph)
    return System(
        elements=[GroupElement('walkers', element=NetworkWalkerElement, size=n_walkers)],
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
        >>> network_diffusion(nx.path_graph(4)).elements[0].check().prepare().size_
        4
    """
    graph = _graph_or_default(graph)
    initial = np.zeros(graph.number_of_nodes())
    initial[source] = 1.0
    nodes = FieldElement('nodes', element=NodeContentElement, space=NetworkSpace(graph), values={'u': initial, 'D': D})
    return System(elements=[nodes])


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
        >>> pagerank(nx.cycle_graph(5)).elements[0].check().prepare().size_
        5
    """
    graph = _graph_or_default(graph)
    size = graph.number_of_nodes()
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    values = {'rank': np.full(size, 1 / size), 'degree': degrees, 'n_nodes': float(size), 'damping': damping}
    return System(elements=[FieldElement('pages', element=NodeRankElement, space=NetworkSpace(graph), values=values)])
