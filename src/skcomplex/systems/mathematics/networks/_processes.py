"""Network systems: random walks, diffusion, and PageRank on graphs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement, GroupElement
from ....core.components.interactions import CouplingInteraction, DiffusionInteraction, MovementInteraction
from ....core.components.spaces import NetworkSpace
from ....core.components.system import BaseSystem, System
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class NodeContentElement(Element):
    """Content of a node of a network, which diffuses along the links with the coefficient `D`.

    Args:
        label:
            The label of the node or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial content.

        D:
            The diffusion coefficient of every link.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import NodeContentElement
        >>> NodeContentElement('node', D=0.5).D
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.D = D


class NodeRankElement(Element):
    """Page of a network with its PageRank, its degree, the number of pages, and the damping of the surfer.

    Args:
        label:
            The label of the page or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        rank:
            The initial rank.

        damping:
            The probability that the surfer follows a link rather than jumping to a random page.

        degree:
            The number of links of the page.

        n_nodes:
            The number of pages of the network.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import NodeRankElement
        >>> NodeRankElement('page', damping=0.9).damping
        0.9
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rank: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        damping: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.85,
        degree: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        n_nodes: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rank = rank
        self.damping = damping
        self.degree = degree
        self.n_nodes = n_nodes


class NetworkWalkerView(AgentView):
    """View of a walker on a network, with its states as attributes."""

    steps: float


class NetworkWalkerElement(Element):
    """Walker on the nodes of a network, which counts the steps it takes.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        steps:
            The initial number of steps taken.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.mathematics.networks import NetworkWalkerElement
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

    @classmethod
    def _get_view(cls) -> type[NetworkWalkerView]:
        """Return the view of the agents, which types their states."""
        return NetworkWalkerView


def _graph_or_default(graph: nx.Graph | None) -> nx.Graph:
    """Return the graph, or Zachary's karate club when none is given."""
    return nx.karate_club_graph() if graph is None else graph


class NetworkRandomWalkSystem(BaseSystem):
    """Random walkers on a network, which visit every node of a connected graph in proportion to its degree.

    The system has the group `walkers`.

    Args:
        graph:
            The undirected NetworkX graph, or `None` for Zachary's karate club.

        n_walkers:
            The number of walkers, started at random nodes.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.mathematics.networks import NetworkRandomWalkSystem
        >>> NetworkRandomWalkSystem(nx.cycle_graph(6), n_walkers=3).prepare().system_.elements[0].prepare().size_
        3
    """

    field = 'mathematics/networks'
    topics = ('network', 'random walk', 'agents')
    references = ('Thurner et al. 2018', 'Jensen 2023')
    until = 100

    def __init__(
        self,
        graph: nx.Graph | None = None,
        *,
        n_walkers: int = 50,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.graph = graph
        self.n_walkers = n_walkers

    @staticmethod
    def _count_step(agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Count a step the walker takes."""
        if isinstance(agent, NetworkWalkerView):
            agent.steps += 1

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        graph = _graph_or_default(self.graph)
        walk = MovementInteraction('walk', target='walkers', step='neighbor', cost=self._count_step)
        return System(
            self.label,
            elements=[GroupElement('walkers', element=NetworkWalkerElement, size=self.n_walkers)],
            interactions=[walk],
            space=NetworkSpace(graph),
        )


class NetworkDiffusionSystem(BaseSystem):
    """Diffusion on a network, `du_i/dt = D Σ_j A_ij (u_j - u_i)`, which spreads a unit mass to every node.

    The total is conserved, and on a connected graph the content of every node tends to `1 / n`. The system has the
    contents `nodes.u`.

    Args:
        graph:
            The undirected NetworkX graph, or `None` for Zachary's karate club.

        D:
            The diffusion coefficient of every link.

        source:
            The index, in node order, of the node that holds the whole mass at the start.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.mathematics.networks import NetworkDiffusionSystem
        >>> NetworkDiffusionSystem(nx.path_graph(4)).prepare().system_.elements[0].prepare().size_
        4
    """

    field = 'mathematics/networks'
    topics = ('network', 'diffusion', 'graph laplacian', 'ode')
    references = ('Thurner et al. 2018', 'Jensen 2023')
    until = 10.0
    dt = 0.1

    def __init__(
        self,
        graph: nx.Graph | None = None,
        *,
        D: float = 1.0,
        source: int = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.graph = graph
        self.D = D
        self.source = source

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        graph = _graph_or_default(self.graph)
        initial = np.zeros(graph.number_of_nodes())
        initial[self.source] = 1.0
        nodes = FieldElement(
            'nodes',
            element=NodeContentElement,
            space=NetworkSpace(graph),
            values={'u': initial, 'D': self.D},
        )
        diffusion = DiffusionInteraction('diffusion', target='nodes', state='u', coefficient=lambda s: s.target.D)
        return System(self.label, elements=[nodes], interactions=[diffusion])


class PagerankSystem(BaseSystem):
    """The PageRank iteration `r_i' = (1 - d) / n + d Σ_j A_ij r_j / k_j` on an undirected network.

    The ranks start uniform, stay normalized, and converge to the PageRank of the graph. Every node needs a link. The
    system has the ranks `pages.rank`.

    Args:
        graph:
            The undirected NetworkX graph without isolated nodes, or `None` for Zachary's karate club.

        damping:
            The probability `d` that the surfer follows a link rather than jumping to a random node.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.mathematics.networks import PagerankSystem
        >>> PagerankSystem(nx.cycle_graph(5)).prepare().system_.elements[0].prepare().size_
        5
    """

    field = 'mathematics/networks'
    topics = ('network', 'pagerank', 'centrality', 'discrete map')
    references = ('Brin and Page 1998', 'Thurner et al. 2018', 'Jensen 2023')
    until = 50
    dt = 1

    def __init__(
        self,
        graph: nx.Graph | None = None,
        *,
        damping: float = 0.85,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.graph = graph
        self.damping = damping

    @staticmethod
    def _get_link_term(s: SimpleNamespace) -> object:
        """Return what one link brings to the rank of a page: a share of the rank of the other page and of the jumps."""
        member, other = s.member, s.other
        return member.damping * other.rank / other.degree + (1 - member.damping) / (member.n_nodes * member.degree)

    def _build_system(self) -> System:
        """Return the PageRank iteration `r_i' = (1 - d) / n + d Σ_j A_ij r_j / k_j` on an undirected network."""
        graph = _graph_or_default(self.graph)
        size = graph.number_of_nodes()
        degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
        values = {'rank': np.full(size, 1 / size), 'degree': degrees, 'n_nodes': float(size), 'damping': self.damping}
        pages = FieldElement('pages', element=NodeRankElement, space=NetworkSpace(graph), values=values)
        links = CouplingInteraction(
            'links',
            target='pages',
            read='rank',
            write='rank',
            function=self._get_link_term,
            form='map',
        )
        return System(self.label, elements=[pages], interactions=[links])
