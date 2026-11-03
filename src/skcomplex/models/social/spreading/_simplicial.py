"""Simplicial contagion: infection through the links and the filled triangles of a simplicial complex."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import networkx as nx
import numpy as np
import xgi

from ....core.elements import LatticeElement
from ....core.spaces import NetworkSpace
from ....core.system import System
from ....specs import ObservableSpec, VisualSpec
from ..._catalog import register_model
from ._infective import InfectiveElement
from ._simplicial_infection import PAIR, SimplicialInfectionInteraction

TRIANGLE = 3


def _simplicial_complex(n_nodes: int, links: float, triangles: float, seed: int) -> tuple[nx.Graph, list[tuple]]:
    """Return a random simplicial complex: random links, and random filled triangles whose sides join the links."""
    simplices = xgi.random_simplicial_complex(n_nodes, [links, triangles], seed=seed)
    members = [tuple(sorted(edge)) for edge in simplices.edges.members(dtype=list)]
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(edge for edge in members if len(edge) == PAIR)
    return graph, [*graph.edges, *(edge for edge in members if len(edge) == TRIANGLE)]


@register_model(
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
            The model, with the states `nodes.infected` and the prevalence `prevalence`.

    Examples:
        >>> from skcomplex.models.social.spreading import simplicial_contagion
        >>> system = simplicial_contagion(30, mean_degree=6.0, mean_triangles=2.0)
        >>> system.elements[0].check().prepare().size_
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
