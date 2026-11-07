"""Simplicial contagion: infection through the links and the filled triangles of a simplicial complex."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import xgi
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, LatticeElement
from ....core.components.interactions import HyperedgeInteraction
from ....core.components.laws import BaseLaw, JumpLaw
from ....core.components.laws.utils import where
from ....core.components.spaces import NetworkSpace
from ....core.components.system import System
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system


class InfectiveElement(Element):
    """Node of a contagion, susceptible at 0 and infected at 1, that recovers at a rate.

    Args:
        label:
            The label of the node or the group.

        visual:
            How the node is drawn, or `None` for `default_visual_`.

        infected:
            The initial state, 0 or 1.

        recovery:
            The rate at which an infected node recovers.

    Examples:
        >>> from skcomplex.systems.social.spreading import InfectiveElement
        >>> InfectiveElement('node').recovery
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        infected: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        recovery: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.infected = infected
        self.recovery = recovery

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the recovery of an infected node."""
        return [JumpLaw(s.recovery * where(s.infected == 1, 1, 0), {s.infected: -1})]


PAIR = 2


class SimplicialInfectionInteraction(HyperedgeInteraction):
    """Infection of a susceptible node by the other members of a link or a filled triangle, when all are infected.

    A susceptible node of a link is infected at rate `beta` when its neighbour is infected, and a susceptible node of a
    triangle at rate `beta_triangle` when both other members are infected.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of nodes.

        beta:
            The rate of infection through a link.

        beta_triangle:
            The rate of infection through a triangle.

    Examples:
        >>> from skcomplex.systems.social.spreading import SimplicialInfectionInteraction
        >>> SimplicialInfectionInteraction('infection', target='nodes', beta_triangle=0.5).beta_triangle
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        beta_triangle: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.beta = beta
        self.beta_triangle = beta_triangle

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the infection of the focal node when every other member of its link or triangle is infected."""
        infected = s.member.infected
        rate = s.beta if len(s.others) + 1 == PAIR else s.beta_triangle
        exposed = math.prod((other.infected for other in s.others), start=where(infected == 0, 1, 0))
        return [JumpLaw(rate * exposed, {infected: 1})]


TRIANGLE = 3


def _simplicial_complex(n_nodes: int, links: float, triangles: float, seed: int) -> tuple[nx.Graph, list[tuple]]:
    """Return a random simplicial complex: random links, and random filled triangles whose sides join the links."""
    simplices = xgi.random_simplicial_complex(n_nodes, [links, triangles], seed=seed)
    members = [tuple(sorted(edge)) for edge in simplices.edges.members(dtype=list)]
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(edge for edge in members if len(edge) == PAIR)
    return graph, [*graph.edges, *(edge for edge in members if len(edge) == TRIANGLE)]


@register_system(
    field='social/spreading',
    tags=('contagion', 'network', 'hypergraph', 'simplicial complex', 'bistability'),
    references=('Iacopini et al. 2019', 'Majhi et al. 2022'),
    defaults={'until': 20.0, 'dt': 0.5},
    observables=lambda system: [ObservableSpec('prevalence', group='nodes', variable='infected')],
)
def simplicial_contagion(
    n_nodes: int = 200,
    *,
    infectivity: float = 0.75,
    triangle_infectivity: float = 3.0,
    mean_degree: float = 20.0,
    mean_triangles: float = 6.0,
    recovery: float = 1.0,
    share: float = 0.4,
    seed: int = 0,
) -> System:
    """Return the simplicial contagion of Iacopini et al., an SIS process on the links and triangles of a complex.

    A random simplicial complex joins every pair of nodes with probability `p1` and fills every triple with probability
    `p2`, chosen so that a node has `mean_degree` links and `mean_triangles` triangles on average. A susceptible node is
    infected at rate `beta = infectivity * recovery / <k>` by every infected neighbour, and at rate `beta_triangle =
    triangle_infectivity * recovery / <k_triangle>` by every triangle whose two other members are infected, and an
    infected node recovers at rate `recovery`. The mean-field prevalence follows `rho' = -rho + infectivity rho (1 -
    rho) + triangle_infectivity rho^2 (1 - rho)` in units of the recovery time. For `triangle_infectivity > 1` and
    infectivities below the epidemic threshold 1, an endemic state coexists with the healthy one, so the epidemic
    dies out from a small seed and persists from a large one.

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

    Returns:
        system:
            The system, with the states `nodes.infected` and the prevalence `prevalence`.

    Examples:
        >>> from skcomplex.systems.social.spreading import simplicial_contagion
        >>> system = simplicial_contagion(30, mean_degree=6.0, mean_triangles=2.0)
        >>> system.elements[0].prepare().size_
        30
    """
    pairs = math.comb(n_nodes - 1, PAIR - 1)
    triples = math.comb(n_nodes - 1, TRIANGLE - 1)
    triangles = min(mean_triangles / triples, 1.0)
    links = min(max(mean_degree - 2 * mean_triangles, 0.0) / pairs, 1.0)
    graph, hyperedges = _simplicial_complex(n_nodes, links, triangles, seed)
    sizes = [len(members) for members in hyperedges]
    degree = max(PAIR * sizes.count(PAIR) / n_nodes, 1.0)
    shared = max(TRIANGLE * sizes.count(TRIANGLE) / n_nodes, 1.0)
    rng = np.random.default_rng(seed + 1)
    infected = (rng.permutation(n_nodes) < round(share * n_nodes)).astype(float)
    nodes = LatticeElement(
        'nodes',
        element=InfectiveElement,
        space=NetworkSpace(graph, hyperedges=hyperedges),
        values={'infected': infected, 'recovery': recovery},
        visual=VisualSpec('sites', {'states': ['infected'], 'low': 0.0, 'high': 1.0}),
    )
    infection = SimplicialInfectionInteraction(
        'infection',
        target='nodes',
        beta=infectivity * recovery / degree,
        beta_triangle=triangle_infectivity * recovery / shared,
    )
    return System(elements=[nodes], interactions=[infection])
