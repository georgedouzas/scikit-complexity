"""Measure networks: degree distributions, clustering, path lengths, components, and centralities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt
from sklearn.metrics import normalized_mutual_info_score

from ..base import AnalysisError
from ..space import NetworkSpace
from ._kinds import DataKind, analysis
from ._statistics import fit_power_law

CENTRALITIES = ('degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank')
COMMUNITY_METHODS = ('louvain', 'label_propagation')
MIN_FIT_NODES = 50
HEAVY_TAIL = 5
FIT_CANDIDATES = 20


def as_graph(network: object) -> nx.Graph:
    """Return the NetworkX graph of a graph or a network space.

    Args:
        network:
            A NetworkX graph or a network space.

    Returns:
        graph:
            The graph.

    Raises:
        AnalysisError:
            The object is neither.
    """
    if isinstance(network, NetworkSpace):
        return network.graph
    if isinstance(network, nx.Graph):
        return network
    error_msg = f'A network is a NetworkX graph or a network space, not {type(network).__name__}.'
    raise AnalysisError(error_msg)


@dataclass(frozen=True)
class NetworkSummary:
    """Summary measures of a network.

    Args:
        nodes:
            The number of nodes.

        edges:
            The number of edges.

        mean_degree:
            The mean degree.

        clustering:
            The average clustering coefficient.

        path_length:
            The average shortest path length in the largest component.

        diameter:
            The longest shortest path in the largest component.

        components:
            The sizes of the connected components, largest first.
    """

    nodes: int
    edges: int
    mean_degree: float
    clustering: float
    path_length: float
    diameter: int
    components: list[int]


@analysis(DataKind.NETWORK)
def degrees(network: object) -> npt.NDArray[np.int64]:
    """Return the degree of every node, in node order.

    Args:
        network:
            A NetworkX graph or a network space.

    Returns:
        degrees:
            The degrees.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.analysis import degrees
        >>> degrees(nx.star_graph(3)).tolist()
        [3, 1, 1, 1]
    """
    return np.asarray([degree for _, degree in as_graph(network).degree()], dtype=int)


@analysis(DataKind.NETWORK)
def summarize_network(network: object) -> NetworkSummary:
    """Return the size, mean degree, clustering, path length, diameter, and components of a network.

    Args:
        network:
            A NetworkX graph or a network space.

    Returns:
        summary:
            The summary measures, with path measures computed on the largest component.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.analysis import summarize_network
        >>> summary = summarize_network(nx.cycle_graph(10))
        >>> summary.mean_degree, summary.clustering, summary.diameter
        (2.0, 0.0, 5)
    """
    graph = as_graph(network)
    undirected = graph.to_undirected()
    components = sorted((len(component) for component in nx.connected_components(undirected)), reverse=True)
    largest = undirected.subgraph(max(nx.connected_components(undirected), key=len)) if components else undirected
    return NetworkSummary(
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        mean_degree=float(np.mean(degrees(undirected))) if graph.number_of_nodes() else 0.0,
        clustering=float(nx.average_clustering(undirected)) if graph.number_of_nodes() else 0.0,
        path_length=float(nx.average_shortest_path_length(largest)) if largest.number_of_nodes() > 1 else 0.0,
        diameter=int(nx.diameter(largest)) if largest.number_of_nodes() > 1 else 0,
        components=components,
    )


@analysis(DataKind.NETWORK)
def centrality(network: object, kind: str = 'degree') -> dict[object, float]:
    """Return the centrality of every node: degree, betweenness, closeness, eigenvector, or PageRank.

    Args:
        network:
            A NetworkX graph or a network space.

        kind:
            The centrality.

    Returns:
        centrality:
            A mapping from node to its centrality.

    Raises:
        AnalysisError:
            The centrality is unknown.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.analysis import centrality
        >>> centrality(nx.star_graph(3), 'betweenness')[0]
        1.0
    """
    functions = {
        'degree': nx.degree_centrality,
        'betweenness': nx.betweenness_centrality,
        'closeness': nx.closeness_centrality,
        'eigenvector': nx.eigenvector_centrality_numpy,
        'pagerank': nx.pagerank,
    }
    if kind not in functions:
        error_msg = f'Unknown centrality {kind!r}. Use one of {", ".join(CENTRALITIES)}.'
        raise AnalysisError(error_msg)
    return dict(functions[kind](as_graph(network)))


@analysis(DataKind.NETWORK)
def degree_moments(network: object, orders: Sequence[int] = (1, 2)) -> list[float]:
    """Return the moments `<k^n>` of the degree distribution of a network.

    Args:
        network:
            A NetworkX graph or a network space.

        orders:
            The orders `n` of the moments.

    Returns:
        moments:
            One moment per order.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.analysis import degree_moments
        >>> degree_moments(nx.star_graph(3))
        [1.5, 3.0]
    """
    values = degrees(network).astype(float)
    return [float(np.mean(values**order)) for order in orders]


@analysis(DataKind.NETWORK)
def assortativity(network: object, attribute: str | None = None, *, numeric: bool = False) -> float:
    """Return the assortativity of a network by degree, or by a node attribute.

    Args:
        network:
            A NetworkX graph or a network space.

        attribute:
            The node attribute, or `None` for the degree.

        numeric:
            Whether the attribute is a number compared by correlation rather than a category compared by equality.

    Returns:
        assortativity:
            Positive when linked nodes are alike, 0 for random mixing, and negative when unlike nodes link.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.analysis import assortativity
        >>> graph = nx.Graph([(0, 1), (2, 3)])
        >>> nx.set_node_attributes(graph, {0: 'a', 1: 'a', 2: 'b', 3: 'b'}, 'group')
        >>> assortativity(graph, 'group')
        1.0
    """
    graph = as_graph(network)
    if attribute is None:
        return float(nx.degree_assortativity_coefficient(graph))
    if numeric:
        return float(nx.numeric_assortativity_coefficient(graph, attribute))
    return float(nx.attribute_assortativity_coefficient(graph, attribute))


@analysis(DataKind.NETWORK)
def neighbor_degree(network: object) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Return the mean degree of the neighbours of nodes against their degree, rising in assortative networks.

    Args:
        network:
            A NetworkX graph or a network space.

    Returns:
        connectivity:
            The distinct degrees and the mean neighbour degree of the nodes of every degree.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.analysis import neighbor_degree
        >>> found = neighbor_degree(nx.star_graph(3))
        >>> found[0].tolist(), found[1].tolist()
        ([1, 3], [3.0, 1.0])
    """
    connectivity = nx.average_degree_connectivity(as_graph(network))
    ordered = sorted(connectivity)
    return np.asarray(ordered, dtype=np.int64), np.asarray([connectivity[degree] for degree in ordered], dtype=float)


@dataclass(frozen=True)
class Communities:
    """Communities of a network.

    Args:
        labels:
            The community of every node, in node order.

        modularity:
            The modularity of the partition.

        agreement:
            The normalized mutual information with the reference partition, or `None` without one.
    """

    labels: npt.NDArray[np.int64]
    modularity: float
    agreement: float | None


@analysis(DataKind.NETWORK)
def communities(
    network: object,
    method: str = 'louvain',
    *,
    reference: Sequence[object] | str | None = None,
    random_state: int | None = None,
) -> Communities:
    """Return the communities of a network by modularity maximization or label propagation, with their agreement.

    Args:
        network:
            A NetworkX graph or a network space.

        method:
            `'louvain'` for Louvain modularity maximization, or `'label_propagation'`.

        reference:
            The reference community of every node in node order, the name of a node attribute that holds it, or
            `None`.

        random_state:
            The seed of the Louvain method.

    Returns:
        communities:
            The labels, the modularity, and the agreement with the reference.

    Raises:
        AnalysisError:
            The method is unknown.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.analysis import communities
        >>> graph = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
        >>> found = communities(graph, reference=[0] * 5 + [1] * 5, random_state=0)
        >>> round(found.modularity, 2), found.agreement
        (0.5, 1.0)
    """
    graph = as_graph(network)
    if method not in COMMUNITY_METHODS:
        error_msg = f'Unknown method {method!r}. Use one of {", ".join(COMMUNITY_METHODS)}.'
        raise AnalysisError(error_msg)
    if method == 'louvain':
        partition = nx.community.louvain_communities(graph, seed=random_state)
    else:
        partition = list(nx.community.label_propagation_communities(graph.to_undirected()))
    membership = {node: index for index, members in enumerate(partition) for node in members}
    labels = np.asarray([membership[node] for node in graph.nodes()], dtype=np.int64)
    modularity = float(nx.community.modularity(graph, partition))
    if reference is None:
        return Communities(labels, modularity, None)
    truth = [graph.nodes[node][reference] for node in graph.nodes()] if isinstance(reference, str) else reference
    return Communities(labels, modularity, float(normalized_mutual_info_score(list(truth), labels)))


def _degree_exponent(graph: nx.Graph) -> float:
    """Return the fitted exponent of a heavy-tailed degree distribution, or `nan` for a small or narrow one."""
    values = degrees(graph)
    if values.size < MIN_FIT_NODES or values.max() < HEAVY_TAIL * max(values.mean(), 1.0):
        return float('nan')
    return fit_power_law(values[values > 0], discrete=True, bootstrap=0, candidates=FIT_CANDIDATES).alpha


def _safe_assortativity(graph: nx.Graph) -> float:
    """Return the degree assortativity, or `nan` when every node has the same degree."""
    values = degrees(graph)
    if values.size < 2 or values.min() == values.max():  # noqa: PLR2004
        return float('nan')
    return float(nx.degree_assortativity_coefficient(graph))


@analysis(DataKind.NETWORK)
def network_frames(
    graphs: Sequence[object],
    measures: Mapping[str, Callable[[nx.Graph], float]] | None = None,
) -> list[dict[str, float]]:
    """Return network measures of every frame of an evolving network, one row per frame.

    The default measures are the numbers of nodes and edges, the mean degree, the clustering, the degree
    assortativity, and the fitted degree exponent where the degree distribution has a heavy tail.

    Args:
        graphs:
            The graph or network space of every frame.

        measures:
            A mapping from name to a function of a graph, added to the default measures.

    Returns:
        rows:
            One row per frame.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.analysis import network_frames
        >>> rows = network_frames([nx.path_graph(3), nx.complete_graph(3)])
        >>> [row['clustering'] for row in rows]
        [0.0, 1.0]
    """
    rows = []
    for network in graphs:
        graph = as_graph(network)
        row = {
            'nodes': float(graph.number_of_nodes()),
            'edges': float(graph.number_of_edges()),
            'mean_degree': float(degrees(graph).mean()) if graph.number_of_nodes() else 0.0,
            'clustering': float(nx.average_clustering(graph)) if graph.number_of_nodes() else 0.0,
            'assortativity': _safe_assortativity(graph),
            'degree_exponent': _degree_exponent(graph),
        }
        row.update({name: float(function(graph)) for name, function in (measures or {}).items()})
        rows.append(row)
    return rows
