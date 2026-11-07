"""Measure networks: degrees, mixing, centralities, communities, neighbour degrees, and summary measures."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt
from sklearn.metrics import normalized_mutual_info_score

from ..specs import AnalysisTags
from ..types import Array, DataKind
from ._base import _check_choice, _check_settings, _check_tags, _read_networks
from ._fit import fit_law
from .types import Source

KINDS = ('degrees', 'assortativity', 'centrality', 'communities', 'neighbor-degree', 'summary')
CENTRALITIES = ('degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank')
METHODS = ('louvain', 'label_propagation')
SETTINGS = {
    'degrees': {'orders'},
    'assortativity': {'attribute', 'numeric'},
    'centrality': {'measure'},
    'communities': {'method', 'reference', 'random_state'},
    'neighbor-degree': set(),
    'summary': set(),
}
"""Settings every kind reads, besides `kind`."""

MIN_FIT_NODES = 50
HEAVY_TAIL = 5
FIT_CANDIDATES = 20

TAGS = AnalysisTags(DataKind.NETWORK, space=frozenset({'network'}))


@dataclass(frozen=True)
class NetworkMeasuresResult:
    """Measures of the network of every frame of a trajectory, or of a graph.

    Args:
        graphs:
            The graph of every frame.

        degrees:
            The degree of every node of every frame, for `'degrees'`.

        moments:
            The degree moments, frames by orders, for `'degrees'`.

        exponent:
            The fitted degree exponent of every frame, for `'degrees'`.

        assortativity:
            The assortativity of every frame, for `'assortativity'`.

        centrality:
            A mapping from node to its centrality for every frame, for `'centrality'`.

        labels:
            The community of every node of every frame, for `'communities'`.

        modularity:
            The modularity of every frame, for `'communities'`.

        agreement:
            The normalized mutual information with the reference of every frame, or `None` without a reference.

        distinct_degrees:
            The distinct degrees of every frame, for `'neighbor-degree'`.

        neighbor_degree:
            The mean neighbour degree of the nodes of every distinct degree of every frame, for `'neighbor-degree'`.

        nodes:
            The number of nodes of every frame, for `'summary'`.

        edges:
            The number of edges of every frame, for `'summary'`.

        mean_degree:
            The mean degree of every frame, for `'summary'`.

        clustering:
            The average clustering of every frame, for `'summary'`.

        path_length:
            The average shortest path length of the largest component of every frame, for `'summary'`.

        diameter:
            The diameter of the largest component of every frame, for `'summary'`.

        components:
            The sizes of the connected components of every frame, largest first, for `'summary'`.
    """

    graphs: list[nx.Graph]
    degrees: list[npt.NDArray[np.int64]] | None = None
    moments: Array | None = None
    exponent: Array | None = None
    assortativity: Array | None = None
    centrality: list[dict[object, float]] | None = None
    labels: list[npt.NDArray[np.int64]] | None = None
    modularity: Array | None = None
    agreement: Array | None = None
    distinct_degrees: list[npt.NDArray[np.int64]] | None = None
    neighbor_degree: list[npt.NDArray[np.float64]] | None = None
    nodes: npt.NDArray[np.int64] | None = None
    edges: npt.NDArray[np.int64] | None = None
    mean_degree: Array | None = None
    clustering: Array | None = None
    path_length: Array | None = None
    diameter: npt.NDArray[np.int64] | None = None
    components: list[list[int]] | None = None


def network_measures(
    source: Source,
    *,
    kind: str = 'summary',
    orders: Sequence[int] = (1, 2),
    attribute: str | None = None,
    numeric: bool = False,
    measure: str = 'degree',
    method: str = 'louvain',
    reference: Sequence[object] | str | None = None,
    random_state: int | None = None,
) -> NetworkMeasuresResult:
    """Return measures of the network of every frame of a trajectory, or of a graph, of one kind.

    The kinds read the graph of every frame:

    - `'degrees'`: the degree of every node, the moments `<k^n>` of the degree distribution for every order of
      `orders`, and the fitted exponent of a heavy-tailed degree distribution, `nan` below 50 nodes or for a narrow
      one.
    - `'assortativity'`: the assortativity by degree, or by a node attribute, positive when linked nodes are alike.
    - `'centrality'`: the degree, betweenness, closeness, eigenvector, or PageRank centrality of every node.
    - `'communities'`: the communities by Louvain modularity maximization or label propagation, their modularity, and
      their agreement with a reference partition.
    - `'neighbor-degree'`: the mean degree of the neighbours of nodes against their degree, rising in assortative
      networks.
    - `'summary'`: the size, mean degree, clustering, path length, diameter, and components, with the path measures on
      the largest component.

    Args:
        source:
            The trajectory, a graph, or a network snapshot.

        kind:
            The measure to compute, one of the kinds above.

        orders:
            The orders `n` of the degree moments.

        attribute:
            The node attribute of the assortativity, or `None` for the degree.

        numeric:
            Whether the attribute is a number compared by correlation rather than a category compared by equality.

        measure:
            The centrality: `'degree'`, `'betweenness'`, `'closeness'`, `'eigenvector'`, or `'pagerank'`.

        method:
            The community detection: `'louvain'` or `'label_propagation'`.

        reference:
            The reference community of every node in node order, the name of a node attribute that holds it, or
            `None`.

        random_state:
            The seed of the Louvain method.

    Returns:
        result:
            The graphs and the measure of every frame.

    Raises:
        AnalysisError:
            The kind, centrality, or method is unknown, or a setting the kind does not read is changed.

        TagError:
            The system of a trajectory has no network space.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.analysis import network_measures
        >>> summary = network_measures(nx.cycle_graph(10))
        >>> summary.mean_degree.tolist(), summary.clustering.tolist(), summary.diameter.tolist()
        ([2.0], [0.0], [5])
        >>> network_measures(nx.star_graph(3), kind='degrees').moments.tolist()
        [[1.5, 3.0]]
        >>> network_measures(nx.star_graph(3), kind='centrality', measure='betweenness').centrality[0][0]
        1.0
    """
    _check_choice('network_measures', 'kind', kind, KINDS)
    _check_tags(network_measures, source, kind)
    _check_choice('network_measures', 'measure', measure, CENTRALITIES)
    _check_choice('network_measures', 'method', method, METHODS)
    _check_settings(
        network_measures,
        'kind',
        SETTINGS[kind],
        kind=kind,
        orders=orders,
        attribute=attribute,
        numeric=numeric,
        measure=measure,
        method=method,
        reference=reference,
        random_state=random_state,
    )
    graphs = _read_networks(source)
    if kind == 'summary':
        rows = [_summary(graph) for graph in graphs]
        return NetworkMeasuresResult(
            graphs,
            nodes=np.asarray([row['nodes'] for row in rows], dtype=int),
            edges=np.asarray([row['edges'] for row in rows], dtype=int),
            mean_degree=np.asarray([row['mean_degree'] for row in rows], dtype=float),
            clustering=np.asarray([row['clustering'] for row in rows], dtype=float),
            path_length=np.asarray([row['path_length'] for row in rows], dtype=float),
            diameter=np.asarray([row['diameter'] for row in rows], dtype=int),
            components=[row['components'] for row in rows if isinstance(row['components'], list)],
        )
    if kind == 'degrees':
        degrees = [_degrees(graph) for graph in graphs]
        return NetworkMeasuresResult(
            graphs,
            degrees=degrees,
            moments=np.asarray(
                [[float(np.mean(values.astype(float) ** order)) for order in orders] for values in degrees],
            ),
            exponent=np.asarray([_degree_exponent(graph) for graph in graphs]),
        )
    if kind == 'assortativity':
        assortativity = np.asarray([_assortativity(graph, attribute, numeric=numeric) for graph in graphs])
        return NetworkMeasuresResult(graphs, assortativity=assortativity)
    if kind == 'centrality':
        return NetworkMeasuresResult(graphs, centrality=[_centrality(graph, measure) for graph in graphs])
    if kind == 'neighbor-degree':
        pairs = [_neighbor_degree(graph) for graph in graphs]
        return NetworkMeasuresResult(
            graphs,
            distinct_degrees=[degrees for degrees, _ in pairs],
            neighbor_degree=[connectivity for _, connectivity in pairs],
        )
    found = [_communities(graph, method, reference, random_state) for graph in graphs]
    return NetworkMeasuresResult(
        graphs,
        labels=[labels for labels, _, _ in found],
        modularity=np.asarray([modularity for _, modularity, _ in found]),
        agreement=None if reference is None else np.asarray([agreement for _, _, agreement in found]),
    )


def _degrees(graph: nx.Graph) -> npt.NDArray[np.int64]:
    """Return the degree of every node, in node order."""
    return np.asarray([degree for _, degree in graph.degree()], dtype=int)


def _degree_exponent(graph: nx.Graph) -> float:
    """Return the fitted exponent of a heavy-tailed degree distribution, or `nan` for a small or narrow one."""
    values = _degrees(graph)
    if values.size < MIN_FIT_NODES or values.max() < HEAVY_TAIL * max(values.mean(), 1.0):
        return float('nan')
    exponent = fit_law(
        values[values > 0],
        discrete=True,
        bootstrap=0,
        candidates=FIT_CANDIDATES,
        compare_with=(),
    ).exponent
    return float('nan') if exponent is None else exponent


def _summary(graph: nx.Graph) -> dict[str, float | int | list[int]]:
    """Return the size, mean degree, clustering, path length, diameter, and components of a network."""
    undirected = graph.to_undirected()
    components = sorted((len(component) for component in nx.connected_components(undirected)), reverse=True)
    largest = undirected.subgraph(max(nx.connected_components(undirected), key=len)) if components else undirected
    return {
        'nodes': graph.number_of_nodes(),
        'edges': graph.number_of_edges(),
        'mean_degree': float(np.mean(_degrees(undirected))) if graph.number_of_nodes() else 0.0,
        'clustering': float(nx.average_clustering(undirected)) if graph.number_of_nodes() else 0.0,
        'path_length': float(nx.average_shortest_path_length(largest)) if largest.number_of_nodes() > 1 else 0.0,
        'diameter': int(nx.diameter(largest)) if largest.number_of_nodes() > 1 else 0,
        'components': components,
    }


def _assortativity(graph: nx.Graph, attribute: str | None, *, numeric: bool) -> float:
    """Return the assortativity of a network by degree, or by a categorical or numeric node attribute."""
    if attribute is None:
        return float(nx.degree_assortativity_coefficient(graph))
    if numeric:
        return float(nx.numeric_assortativity_coefficient(graph, attribute))
    return float(nx.attribute_assortativity_coefficient(graph, attribute))


def _centrality(graph: nx.Graph, measure: str) -> dict[object, float]:
    """Return the degree, betweenness, closeness, eigenvector, or PageRank centrality of every node."""
    functions = {
        'degree': nx.degree_centrality,
        'betweenness': nx.betweenness_centrality,
        'closeness': nx.closeness_centrality,
        'eigenvector': nx.eigenvector_centrality_numpy,
        'pagerank': nx.pagerank,
    }
    return dict(functions[measure](graph))


def _neighbor_degree(graph: nx.Graph) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Return the distinct degrees and the mean degree of the neighbours of the nodes of every degree."""
    connectivity = nx.average_degree_connectivity(graph)
    ordered = sorted(connectivity)
    return np.asarray(ordered, dtype=np.int64), np.asarray(
        [connectivity[degree] for degree in ordered],
        dtype=float,
    )


def _communities(
    graph: nx.Graph,
    method: str,
    reference: Sequence[object] | str | None,
    random_state: int | None,
) -> tuple[npt.NDArray[np.int64], float, float | None]:
    """Return the community of every node, the modularity, and the agreement with a reference partition."""
    if method == 'louvain':
        partition = nx.community.louvain_communities(graph, seed=random_state)
    else:
        partition = list(nx.community.label_propagation_communities(graph.to_undirected()))
    membership = {node: index for index, members in enumerate(partition) for node in members}
    labels = np.asarray([membership[node] for node in graph.nodes()], dtype=np.int64)
    modularity = float(nx.community.modularity(graph, partition))
    if reference is None:
        return labels, modularity, None
    truth = [graph.nodes[node][reference] for node in graph.nodes()] if isinstance(reference, str) else reference
    return labels, modularity, float(normalized_mutual_info_score(list(truth), labels))
