"""Measure networks: degree distributions, clustering, path lengths, components, and centralities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt

from ..core import AnalysisError
from ..model import NetworkSpace

CENTRALITIES = ('degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank')


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
        >>> from skcomplex.analysis import degrees
        >>> degrees(nx.star_graph(3)).tolist()
        [3, 1, 1, 1]
    """
    return np.asarray([degree for _, degree in as_graph(network).degree()], dtype=int)


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
        >>> from skcomplex.analysis import summarize_network
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
        >>> from skcomplex.analysis import centrality
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
