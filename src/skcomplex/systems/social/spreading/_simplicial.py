"""Simplicial contagion: infection through a simplicial complex.

The infection goes through the links and the filled triangles.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import xgi
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import JumpLaw
from ....core.components.laws.utils import where
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Role

PAIR = 2
TRIANGLE = 3


class InfectiveElement(Element):
    """Nodes of a contagion, susceptible at 0 and infected at 1.

    They recover at a rate, by a `RecoveryInteraction`.

    Args:
        label:
            The label of the node or the group.

        infected:
            The initial state, 0 or 1, one per node or shared.

        recovery:
            The rate at which an infected node recovers.

        space:
            The network with one node per site, or `None` for one node.

    Examples:
        >>> from skcomplex.systems.social.spreading import InfectiveElement
        >>> InfectiveElement('node').recovery
        1.0
    """

    visual = VisualSpec('sites', {'states': ['infected'], 'low': 0.0, 'high': 1.0})

    def __init__(
        self,
        label: str,
        *,
        infected: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0.0,
        recovery: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.infected = infected
        self.recovery = recovery


class RecoveryInteraction(Interaction):
    """Recovery of an infected node at the rate `recovery` of the nodes.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of nodes.

    Examples:
        >>> from skcomplex.systems.social.spreading import RecoveryInteraction
        >>> RecoveryInteraction('recovery', target='nodes').target
        'nodes'
    """

    law_classes = (JumpLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the recovery of an infected node as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            node = context.target
            self.laws_: list[BaseComponent] = [
                JumpLaw(node.recovery * where(node.infected == 1, 1, 0), {node.infected: -1}),
            ]


class SimplicialInfectionInteraction(Interaction):
    """Infection of a susceptible node by the other members of a link or a triangle.

    The triangle is filled, and the infection needs all of them infected. A susceptible node of a link is infected at
    rate `beta` when its neighbour is infected, and a susceptible node of a triangle at rate `beta_triangle` when both
    other members are infected. The laws run once for every link and triangle of the network.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        among:
            The path of the group of nodes.

        beta:
            The rate of infection through a link.

        beta_triangle:
            The rate of infection through a triangle.

        along:
            The links of the network the laws run on, `'hyperedges'` for its links and triangles.

    Examples:
        >>> from skcomplex.systems.social.spreading import SimplicialInfectionInteraction
        >>> SimplicialInfectionInteraction('infection', among='nodes', beta_triangle=0.5).beta_triangle
        0.5
    """

    law_classes = (JumpLaw,)

    def __init__(
        self,
        label: str,
        *,
        among: Role,
        beta: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        beta_triangle: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        along: str = 'hyperedges',
    ) -> None:
        super().__init__(label, among=among, along=along)
        self.beta = beta
        self.beta_triangle = beta_triangle

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the infection of the focal node, when every other member of its link or triangle is infected."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            infected = context.member.infected
            rate = context.beta if len(context.others) + 1 == PAIR else context.beta_triangle
            exposed = math.prod((other.infected for other in context.others), start=where(infected == 0, 1, 0))
            self.laws_: list[BaseComponent] = [JumpLaw(rate * exposed, {infected: 1})]


class SimplicialContagionSystem(BaseSystem):
    """The simplicial contagion of Iacopini et al.

    It is an SIS process on the links and triangles of a complex. A random simplicial complex joins every pair of nodes
    with probability `p1` and fills every triple with probability `p2`, chosen so that a node has `mean_degree` links
    and `mean_triangles` triangles on average. A susceptible node is infected at rate
    `beta = infectivity * recovery / <k>` by every infected neighbour, and at rate
    `beta_triangle = triangle_infectivity * recovery / <k_triangle>` by every triangle whose two other members are
    infected, and an infected node recovers at rate `recovery`. The mean-field prevalence follows
    `rho' = -rho + infectivity rho (1 - rho) + triangle_infectivity rho^2 (1 - rho)` in units of the recovery time. For
    `triangle_infectivity > 1` and infectivities below the epidemic threshold 1, an endemic state coexists with the
    healthy one, so the epidemic dies out from a small seed and persists from a large one.

    The system has the states `nodes.infected` and the prevalence `prevalence`.

    Args:
        n_nodes:
            The number of nodes.

        infectivity:
            The rescaled rate of infection through links.

        triangle_infectivity:
            The rescaled rate of infection through triangles.

        mean_degree:
            The mean number of links of a node, which sets `p1` together with the sides of the triangles.

        mean_triangles:
            The mean number of triangles of a node, which sets `p2`.

        recovery:
            The rate of recovery.

        share:
            The share of infected nodes at the start.

        seed:
            The seed of the complex and the infected nodes.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.spreading import SimplicialContagionSystem
        >>> system = SimplicialContagionSystem(30, mean_degree=6.0, mean_triangles=2.0).prepare().system_
        >>> len(system.elements[0].infected)
        30
    """

    field = 'social/spreading'
    topics = ('contagion', 'network', 'hypergraph', 'simplicial complex', 'bistability')
    references = ('Iacopini et al. 2019', 'Majhi et al. 2022')
    until = 20.0
    dt = 0.5
    observables = (Observable('prevalence', group='nodes', variable='infected'),)

    def __init__(
        self,
        n_nodes: int = 200,
        *,
        infectivity: float = 0.75,
        triangle_infectivity: float = 3.0,
        mean_degree: float = 20.0,
        mean_triangles: float = 6.0,
        recovery: float = 1.0,
        share: float = 0.4,
        seed: int = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_nodes = n_nodes
        self.infectivity = infectivity
        self.triangle_infectivity = triangle_infectivity
        self.mean_degree = mean_degree
        self.mean_triangles = mean_triangles
        self.recovery = recovery
        self.share = share
        self.seed = seed

    @staticmethod
    def _get_simplicial_complex(
        n_nodes: int,
        links: float,
        triangles: float,
        seed: int,
    ) -> tuple[nx.Graph, list[tuple]]:
        """Return a random simplicial complex.

        It holds random links, and random filled triangles whose sides join the links.
        """
        simplices = xgi.random_simplicial_complex(n_nodes, [links, triangles], seed=seed)
        members = [tuple(sorted(edge)) for edge in simplices.edges.members(dtype=list)]
        graph = nx.Graph()
        graph.add_nodes_from(range(n_nodes))
        graph.add_edges_from(edge for edge in members if len(edge) == PAIR)
        return graph, [*graph.edges, *(edge for edge in members if len(edge) == TRIANGLE)]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        pairs = math.comb(self.n_nodes - 1, PAIR - 1)
        triples = math.comb(self.n_nodes - 1, TRIANGLE - 1)
        triangles = min(self.mean_triangles / triples, 1.0)
        links = min(max(self.mean_degree - 2 * self.mean_triangles, 0.0) / pairs, 1.0)
        graph, hyperedges = self._get_simplicial_complex(self.n_nodes, links, triangles, self.seed)
        sizes = [len(members) for members in hyperedges]
        degree = max(PAIR * sizes.count(PAIR) / self.n_nodes, 1.0)
        shared = max(TRIANGLE * sizes.count(TRIANGLE) / self.n_nodes, 1.0)
        rng = np.random.default_rng(self.seed + 1)
        infected = (rng.permutation(self.n_nodes) < round(self.share * self.n_nodes)).astype(float)
        nodes = InfectiveElement(
            'nodes',
            infected=infected,
            recovery=self.recovery,
            space=NetworkSpace(graph, hyperedges=hyperedges),
        )
        infection = SimplicialInfectionInteraction(
            'infection',
            among='nodes',
            beta=self.infectivity * self.recovery / degree,
            beta_triangle=self.triangle_infectivity * self.recovery / shared,
        )
        self.system_ = System(
            self.label,
            elements=[nodes],
            interactions=[RecoveryInteraction('recovery', target='nodes'), infection],
        )
