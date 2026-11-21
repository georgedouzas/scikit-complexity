"""Network systems.

The module covers random walks, diffusion, and PageRank on graphs.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import laplacian, neighbors
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ...general.dynamics import NetworkDynamicalSystem
from ...general.dynamics.types import Equations


class NodeRankElement(Element):
    """Pages of a network, each with its rank `rank` and its number `degree` of links.

    Args:
        label:
            The label of the pages.

        rank:
            The initial rank, one per page or shared.

        degree:
            The number of links, one per page or shared.

        space:
            The network with one page per node, or `None` for one page.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import NodeRankElement
        >>> NodeRankElement('pages', degree=2.0).degree
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        rank: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        degree: Annotated[object, RealSet.open(0, oo)] = 1.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.rank = rank
        self.degree = degree


class PagerankInteraction(Interaction):
    """The PageRank iteration `r_i' = Σ_j A_ij (d r_j / k_j + (1 - d) / (n k_i))` of a random surfer.

    The surfer follows a link with the probability `d` and jumps to a random page of the `n` pages otherwise, so the
    ranks of an undirected network without isolated pages stay normalized.

    Args:
        label:
            The label of the interaction.

        target:
            The pages whose ranks the iteration sets.

        damping:
            The probability `d` that the surfer follows a link rather than jumping to a random page.

        n_nodes:
            The number `n` of pages of the network.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import PagerankInteraction
        >>> PagerankInteraction('links', damping=0.9).damping
        0.9
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role = 'pages', damping: float = 0.85, n_nodes: int = 1) -> None:
        super().__init__(label)
        self.target = target
        self.damping = damping
        self.n_nodes = n_nodes

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the next rank, summed over the links of every page, as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            page, other, damping = context.target, context.target.other, context.damping
            term = damping * other.rank / other.degree + (1 - damping) / (context.n_nodes * page.degree)
            self.laws_: list[BaseLaw] = [UpdateLaw(page.rank, neighbors(term))]


class NetworkWalkerView(AgentViewProtocol, Protocol):
    """View of a walker on a network, with its values as attributes."""

    steps: float


class NetworkWalkerElement(Element):
    """Walkers on the nodes of a network, each with its number `steps` of steps taken.

    Args:
        label:
            The label of the walkers.

        steps:
            The initial number of steps taken, one per walker or shared.

        size:
            The number of walkers, or `None` for one.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import NetworkWalkerElement
        >>> NetworkWalkerElement('walkers', size=5).size
        5
    """

    visual = VisualSpec('agents', {'color_by': 'steps'})

    def __init__(
        self,
        label: str,
        *,
        steps: Annotated[object, NN] = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.steps = steps


class NetworkStepInteraction(Interaction):
    """Step of a walker on a network to a neighbouring node drawn uniformly, counted in `steps`.

    Every walker runs the rule once per step.

    Args:
        label:
            The label of the interaction.

        target:
            The walkers that step.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import NetworkStepInteraction
        >>> NetworkStepInteraction('walk').target
        'walkers'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role = 'walkers') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the step of every walker as the rule."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [UpdateLaw(context.target, self._step)]

    def _step(self, agent: NetworkWalkerView, system: SystemView) -> None:
        """Move a walker to a random neighbouring node and count the step."""
        nodes = agent.get_neighbor_cells()
        if nodes:
            agent.move_to(nodes[int(system.rng.integers(len(nodes)))])
            agent.steps += 1


class NetworkRandomWalkSystem(BaseSystem):
    """Random walkers on a network.

    The walkers visit every node of a connected graph in proportion to its degree.

    The system has the group `walkers`.

    Args:
        graph:
            The undirected NetworkX graph, or `None` for Zachary's karate club.

        n_walkers:
            The number of walkers, started at random nodes.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.mathematics.networks import NetworkRandomWalkSystem
        >>> NetworkRandomWalkSystem(nx.cycle_graph(6), n_walkers=3).prepare().system_.elements[0].size
        3
    """

    field = 'mathematics/networks'
    topics = ('network', 'random walk', 'agents')
    references = ('Thurner et al. 2018', 'Jensen 2023')
    until = 100

    @staticmethod
    def _get_graph_or_default(graph: nx.Graph | None) -> nx.Graph:
        """Return the graph, or Zachary's karate club when none is given."""
        return nx.karate_club_graph() if graph is None else graph

    def __init__(
        self,
        graph: nx.Graph | None = None,
        *,
        n_walkers: int = 50,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.graph = graph
        self.n_walkers = n_walkers

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the walkers on the network and their steps."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[NetworkWalkerElement('walkers', size=self.n_walkers)],
            interactions=[NetworkStepInteraction('walk', target='walkers')],
            space=NetworkSpace(NetworkRandomWalkSystem._get_graph_or_default(self.graph)),
        )


class NetworkDiffusionSystem(NetworkDynamicalSystem):
    """Diffusion on a network, `du_i/dt = D Σ_j A_ij (u_j - u_i)`.

    The diffusion spreads a unit mass to every node.

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
        >>> NetworkDiffusionSystem(nx.path_graph(4)).prepare().simulate(until=0.0).get_values('nodes.u').shape
        (1, 4)
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.graph = graph
        self.D = D
        self.source = source

    def _get_graph(self) -> nx.Graph:
        """Return the graph, Zachary's karate club by default."""
        return NetworkRandomWalkSystem._get_graph_or_default(self.graph)

    def _get_states(self) -> dict[str, object]:
        """Return the content, one at the source and zero elsewhere."""
        initial = np.zeros(self._get_graph().number_of_nodes())
        initial[self.source] = 1.0
        return {'u': initial}

    def _get_parameters(self) -> dict[str, object]:
        """Return the diffusion constant."""
        return {'D': self.D}

    def _get_equations(self) -> Equations:
        """Return the diffusion `D L u` of the content by the graph Laplacian."""
        return {'u': lambda s: s.target.D * laplacian(s.target.u)}


class PagerankSystem(BaseSystem):
    """The PageRank iteration `r_i' = (1 - d) / n + d Σ_j A_ij r_j / k_j`.

    The network is undirected.

    The ranks start uniform, stay normalized, and converge to the PageRank of the graph. Every node needs a link. The
    system has the ranks `pages.rank`.

    Args:
        graph:
            The undirected NetworkX graph without isolated nodes, or `None` for Zachary's karate club.

        damping:
            The probability `d` that the surfer follows a link rather than jumping to a random node.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.mathematics.networks import PagerankSystem
        >>> PagerankSystem(nx.cycle_graph(5)).prepare().system_.elements[0].rank.tolist()
        [0.2, 0.2, 0.2, 0.2, 0.2]
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.graph = graph
        self.damping = damping

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the pages with uniform ranks and their iteration."""
        super()._prepare(context)
        graph = NetworkRandomWalkSystem._get_graph_or_default(self.graph)
        size = graph.number_of_nodes()
        degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
        pages = NodeRankElement('pages', rank=np.full(size, 1 / size), degree=degrees, space=NetworkSpace(graph))
        self.system_ = System(
            self.label,
            elements=[pages],
            interactions=[PagerankInteraction('links', target='pages', damping=self.damping, n_nodes=size)],
        )
