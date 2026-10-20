"""Count network motifs of three and four nodes and score them against degree-preserving null ensembles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections import Counter
from dataclasses import dataclass

import networkx as nx
import numpy as np

from ..base import AnalysisError
from ._kinds import DataKind, analysis
from ._networks import as_graph

NULLS = ('switch', 'configuration')
SIZES = (3, 4)
SWAPS_PER_EDGE = 3
UNDIRECTED_FOUR = {
    (3, (1, 1, 1, 3)): 'star',
    (3, (1, 1, 2, 2)): 'path',
    (4, (2, 2, 2, 2)): 'cycle',
    (4, (1, 2, 2, 3)): 'tailed_triangle',
    (5, (2, 2, 3, 3)): 'diamond',
    (6, (3, 3, 3, 3)): 'clique',
}
PERMUTATIONS = tuple(itertools.permutations(range(4)))


def _extend(graph: nx.Graph, subgraph: set[object], extension: set[object], root: object, size: int) -> list[frozenset]:
    """Return the connected node sets of a size that grow from a set by Wernicke's ESU enumeration."""
    if len(subgraph) == size:
        return [frozenset(subgraph)]
    found = []
    candidates = set(extension)
    order = {node: index for index, node in enumerate(graph.nodes())}
    neighbourhood = {other for node in subgraph for other in graph.neighbors(node)} | subgraph
    while candidates:
        node = candidates.pop()
        exclusive = {
            other for other in graph.neighbors(node) if order[other] > order[root] and other not in neighbourhood
        }
        found.extend(_extend(graph, subgraph | {node}, candidates | exclusive, root, size))
    return found


def _connected_sets(graph: nx.Graph, size: int) -> list[frozenset]:
    """Return every connected set of nodes of a size, once."""
    undirected = graph.to_undirected(as_view=True)
    order = {node: index for index, node in enumerate(undirected.nodes())}
    found = []
    for root in undirected.nodes():
        extension = {other for other in undirected.neighbors(root) if order[other] > order[root]}
        found.extend(_extend(undirected, {root}, extension, root, size))
    return found


def _directed_code(graph: nx.DiGraph, nodes: tuple[object, ...]) -> str:
    """Return a name of the isomorphism class of a directed subgraph of four nodes, its smallest adjacency code."""
    adjacency = nx.to_numpy_array(graph, nodelist=list(nodes), dtype=int)
    codes = []
    for permutation in PERMUTATIONS:
        permuted = adjacency[np.ix_(permutation, permutation)]
        codes.append(int(''.join(map(str, permuted[~np.eye(4, dtype=bool)])), 2))
    return f'directed_{min(codes)}'


def _four_node_name(graph: nx.Graph, nodes: frozenset) -> str:
    """Return the name of the motif a connected set of four nodes induces."""
    subgraph = graph.subgraph(nodes)
    if graph.is_directed():
        return _directed_code(subgraph, tuple(nodes))
    key = (subgraph.number_of_edges(), tuple(sorted(degree for _, degree in subgraph.degree())))
    return UNDIRECTED_FOUR[key]


def _three_node_counts(graph: nx.Graph) -> dict[str, int]:
    """Return the triadic census of a directed graph, or the open and closed triads of an undirected one."""
    if graph.is_directed():
        census = nx.triadic_census(graph)
        return {name: int(count) for name, count in census.items() if name not in {'003', '012', '102'}}
    triangles = sum(nx.triangles(graph).values()) // 3
    wedges = sum(degree * (degree - 1) // 2 for _, degree in graph.degree())
    return {'path': int(wedges - 3 * triangles), 'triangle': int(triangles)}


@analysis(DataKind.NETWORK)
def motif_counts(network: object, size: int = 3) -> dict[str, int]:
    """Return the number of connected induced subgraphs of three or four nodes of every kind in a network.

    Directed triads carry the names of the triadic census, such as `'030T'` for the feed-forward loop. Undirected
    motifs are named by their shape, and directed motifs of four nodes by their smallest adjacency code.

    Args:
        network:
            A NetworkX graph, directed or not, or a network space.

        size:
            The number of nodes, 3 or 4.

    Returns:
        counts:
            The number of occurrences of every motif.

    Raises:
        AnalysisError:
            The size is not 3 or 4.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.analysis import motif_counts
        >>> motif_counts(nx.complete_graph(4))
        {'path': 0, 'triangle': 4}
        >>> motif_counts(nx.complete_graph(4), 4)
        {'clique': 1}
    """
    if size not in SIZES:
        error_msg = f'Motifs have 3 or 4 nodes, not {size}.'
        raise AnalysisError(error_msg)
    graph = as_graph(network)
    if size == 3:  # noqa: PLR2004
        return _three_node_counts(graph)
    return dict(Counter(_four_node_name(graph, nodes) for nodes in _connected_sets(graph, size)))


def _null_graph(graph: nx.Graph, null: str, rng: np.random.Generator) -> nx.Graph:
    """Return a random graph with the degrees of a graph, by edge switches or by the configuration model."""
    seed = int(rng.integers(2**31))
    swaps = SWAPS_PER_EDGE * graph.number_of_edges()
    if null == 'switch' and graph.is_directed():
        return nx.directed_edge_swap(graph.copy(), nswap=swaps, max_tries=100 * swaps, seed=seed)
    if null == 'switch':
        return nx.double_edge_swap(graph.copy(), nswap=swaps, max_tries=100 * swaps, seed=seed)
    if graph.is_directed():
        ins = [degree for _, degree in graph.in_degree()]
        outs = [degree for _, degree in graph.out_degree()]
        multigraph = nx.directed_configuration_model(ins, outs, seed=seed)
        simple = nx.DiGraph(multigraph)
    else:
        simple = nx.Graph(nx.configuration_model([degree for _, degree in graph.degree()], seed=seed))
    simple.remove_edges_from(nx.selfloop_edges(simple))
    return simple


@dataclass(frozen=True)
class MotifScores:
    """Motif counts of a network against a null ensemble.

    Args:
        counts:
            The count of every motif in the network.

        null_mean:
            The mean count of every motif over the null ensemble.

        null_std:
            The standard deviation of every count over the null ensemble.

        zscores:
            `(count - null_mean) / null_std` of every motif, `nan` where the null count never varies.
    """

    counts: dict[str, int]
    null_mean: dict[str, float]
    null_std: dict[str, float]
    zscores: dict[str, float]


@analysis(DataKind.NETWORK)
def motif_zscores(
    network: object,
    size: int = 3,
    *,
    null: str = 'switch',
    samples: int = 100,
    random_state: int | None = None,
) -> MotifScores:
    """Return the z-scores of the motifs of a network against random graphs that keep its degrees.

    Args:
        network:
            A NetworkX graph, directed or not, or a network space.

        size:
            The number of nodes of the motifs, 3 or 4.

        null:
            `'switch'` for degree-preserving edge switches, or `'configuration'` for the configuration model with
            self-loops and multiple edges removed.

        samples:
            The number of random graphs.

        random_state:
            The seed of the ensemble.

    Returns:
        scores:
            The counts, the null means and deviations, and the z-scores.

    Raises:
        AnalysisError:
            The null ensemble or the size is unknown.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.analysis import motif_zscores
        >>> graph = nx.gnm_random_graph(30, 60, seed=0)
        >>> scores = motif_zscores(graph, samples=10, random_state=0)
        >>> sorted(scores.zscores)
        ['path', 'triangle']
    """
    if null not in NULLS:
        error_msg = f'Unknown null ensemble {null!r}. Use one of {", ".join(NULLS)}.'
        raise AnalysisError(error_msg)
    graph = as_graph(network)
    counts = motif_counts(graph, size)
    rng = np.random.default_rng(random_state)
    ensemble = [motif_counts(_null_graph(graph, null, rng), size) for _ in range(samples)]
    names = sorted(set(counts).union(*ensemble))
    table = np.asarray([[sample.get(name, 0) for name in names] for sample in ensemble], dtype=float)
    means, deviations = table.mean(axis=0), table.std(axis=0)
    observed = np.asarray([counts.get(name, 0) for name in names], dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.where(deviations > 0, (observed - means) / deviations, np.nan)
    return MotifScores(
        {name: int(counts.get(name, 0)) for name in names},
        dict(zip(names, means.tolist(), strict=True)),
        dict(zip(names, deviations.tolist(), strict=True)),
        dict(zip(names, scores.tolist(), strict=True)),
    )
