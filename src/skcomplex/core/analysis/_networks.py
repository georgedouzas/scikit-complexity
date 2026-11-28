"""Measure networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
from sklearn.metrics import normalized_mutual_info_score

from ..enums import DataKind
from ..errors import AnalysisError
from ..types import Array
from ._analysis_tags import AnalysisTags
from ._base import _check_choice, _check_settings, _check_tags, _read_networks
from ._fit import fit_law
from ._utils import load_module
from .types import DataSource

KINDS = (
    'degrees',
    'assortativity',
    'centrality',
    'communities',
    'modularity',
    'neighbor-degree',
    'spectrum',
    'summary',
)
CENTRALITIES = ('degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank', 'katz', 'edge_betweenness')
METHODS = ('louvain', 'label_propagation', 'girvan_newman', 'leading_eigenvector')
SPLITTING_METHODS = ('girvan_newman', 'leading_eigenvector')
SUMMARY_MEASURES = ('size', 'clustering', 'transitivity', 'paths', 'components', 'reciprocity')
SETTINGS = {
    'degrees': {'orders'},
    'assortativity': {'attribute', 'numeric'},
    'centrality': {'measure', 'alpha'},
    'communities': {'method', 'reference', 'random_state', 'weight', 'n_communities'},
    'modularity': {'partition', 'weight'},
    'neighbor-degree': set(),
    'spectrum': {'weight'},
    'summary': {'measures', 'samples', 'random_state'},
}
MIN_FIT_NODES = 50
HEAVY_TAIL = 5
FIT_CANDIDATES = 20
KATZ_SHARE = 0.9
TAGS = AnalysisTags(DataKind.NETWORK, space=frozenset({'network'}))


@dataclass(frozen=True)
class NetworkMeasuresResult:
    """A result of `network_measures`, measures of a network at every frame.

    It holds the graph of every frame and what the kind measures: the degrees, how alike linked nodes are, how central
    each node is, the communities, the spectrum, or a summary of size, clustering, and paths, such as the mean degree 2
    of a ring. It reports the hubs of the internet, the clustering of a social network, and the paths of a power grid.

    Read more in the user guide, under Analyse.

    Args:
        graphs:
            The graph of every frame.

        degrees:
            The degree of every node of every frame, for `'degrees'`.

        moments:
            The degree moments, frames by orders, for `'degrees'`.

        exponent:
            The fitted degree exponent of every frame, for `'degrees'`.

        in_degrees:
            The in degree of every node of every frame, `None` for an undirected frame, for `'degrees'`, or `None`
            when no frame is directed.

        out_degrees:
            The out degree of every node of every frame, `None` for an undirected frame, for `'degrees'`, or `None`
            when no frame is directed.

        in_moments:
            The moments of the in degrees of the orders of every frame, `None` for an undirected frame, for
            `'degrees'`, or `None` when no frame is directed.

        out_moments:
            The moments of the out degrees of the orders of every frame, `None` for an undirected frame, for
            `'degrees'`, or `None` when no frame is directed.

        assortativity:
            The assortativity of every frame, for `'assortativity'`.

        centrality:
            A mapping from node to its centrality for every frame, or from edge to its centrality for
            `'edge_betweenness'`, for `'centrality'`.

        labels:
            The community of every node of every frame, from 0, for `'communities'` and `'modularity'`.

        modularity:
            The modularity of every frame, for `'communities'` and `'modularity'`.

        agreement:
            The normalized mutual information with the reference of every frame, or `None` without a reference.

        distinct_degrees:
            The distinct degrees of every frame, for `'neighbor-degree'`.

        neighbor_degree:
            The mean neighbour degree of the nodes of every distinct degree of every frame, for `'neighbor-degree'`.

        adjacency_spectrum:
            The adjacency eigenvalues of every frame, real and ascending for an undirected frame, complex and by
            descending modulus for a directed one, for `'spectrum'`.

        laplacian_spectrum:
            The Laplacian eigenvalues of the undirected graph of every frame, ascending, for `'spectrum'`.

        normalized_laplacian_spectrum:
            The normalised Laplacian eigenvalues of the undirected graph of every frame, ascending, for `'spectrum'`.

        algebraic_connectivity:
            The second smallest Laplacian eigenvalue of every frame, 0 for a disconnected one, for `'spectrum'`.

        fiedler_vector:
            The eigenvector of the algebraic connectivity of every frame in node order, `nan` for a disconnected frame
            and empty below two nodes, for `'spectrum'`.

        spectral_radius:
            The largest modulus of the adjacency eigenvalues of every frame, for `'spectrum'`.

        modularity_matrix:
            The modularity matrix of every frame in node order, for `'spectrum'`.

        nodes:
            The number of nodes of every frame, for `'summary'`.

        edges:
            The number of edges of every frame, for `'summary'`.

        mean_degree:
            The mean degree of every frame, for `'summary'`.

        clustering:
            The average clustering of every frame, for `'summary'`.

        transitivity:
            The share of the connected triples of every frame that close into triangles, for `'summary'`.

        path_length:
            The average shortest path length of the largest component of every frame, estimated from the sampled
            sources when `samples` is set, for `'summary'`.

        diameter:
            The diameter of the largest component of every frame, a lower bound when `samples` is set, for
            `'summary'`.

        components:
            The sizes of the connected components of every frame, largest first, for `'summary'`.

        reciprocity:
            The share of the edges of every frame whose reverse edge is present, `nan` for an undirected frame, for
            `'summary'`.

    See Also:
        network_measures: Returns this result.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.analysis import network_measures
        >>> spectrum = network_measures(nx.cycle_graph(4), kind='spectrum')
        >>> spectrum.laplacian_spectrum[0].round(6).tolist() == [0, 2, 2, 4], spectrum.spectral_radius.round(6).tolist()
        (True, [2.0])
    """

    graphs: list[nx.Graph]
    degrees: list[npt.NDArray[np.int64]] | None = None
    moments: Array | None = None
    exponent: Array | None = None
    in_degrees: list[npt.NDArray[np.int64] | None] | None = None
    out_degrees: list[npt.NDArray[np.int64] | None] | None = None
    in_moments: list[npt.NDArray[np.float64] | None] | None = None
    out_moments: list[npt.NDArray[np.float64] | None] | None = None
    assortativity: Array | None = None
    centrality: list[dict[object, float]] | None = None
    labels: list[npt.NDArray[np.int64]] | None = None
    modularity: Array | None = None
    agreement: Array | None = None
    distinct_degrees: list[npt.NDArray[np.int64]] | None = None
    neighbor_degree: list[npt.NDArray[np.float64]] | None = None
    adjacency_spectrum: list[npt.NDArray[np.float64] | npt.NDArray[np.complex128]] | None = None
    laplacian_spectrum: list[npt.NDArray[np.float64]] | None = None
    normalized_laplacian_spectrum: list[npt.NDArray[np.float64]] | None = None
    algebraic_connectivity: Array | None = None
    fiedler_vector: list[npt.NDArray[np.float64]] | None = None
    spectral_radius: Array | None = None
    modularity_matrix: list[npt.NDArray[np.float64]] | None = None
    nodes: npt.NDArray[np.int64] | None = None
    edges: npt.NDArray[np.int64] | None = None
    mean_degree: Array | None = None
    clustering: Array | None = None
    transitivity: Array | None = None
    path_length: Array | None = None
    diameter: npt.NDArray[np.int64] | None = None
    components: list[list[int]] | None = None
    reciprocity: Array | None = None


def network_measures(
    source: DataSource,
    *,
    kind: str = 'summary',
    orders: Sequence[int] = (1, 2),
    attribute: str | None = None,
    numeric: bool = False,
    measure: str = 'degree',
    alpha: float | None = None,
    method: str = 'louvain',
    n_communities: int | None = None,
    reference: Sequence[object] | str | None = None,
    partition: Sequence[object] | str | None = None,
    weight: str | None = 'weight',
    measures: Sequence[str] | None = None,
    samples: int | None = None,
    random_state: int | None = None,
) -> NetworkMeasuresResult:
    """Return measures of a graph or of the network of every frame of a trajectory.

    A network is a set of nodes joined by links, and the measures describe how many links each node has, how they
    cluster, and how far apart nodes are, such as the mean degree 2, clustering 0, and diameter 5 of a ring of 10
    nodes. The kinds read the graph of every frame. It measures the hubs of the internet, the clustering of a social
    network, the paths of a power grid, and the epidemic threshold `1 / λ1` of a contact network.

    Read more in the user guide, under Analyse.

    Args:
        source:
            The trajectory, a graph, or a network snapshot.

        kind:
            What to return. `'degrees'` gives the degree of every node, the moments `<k^n>` of the degree distribution
            for every order of `orders`, and the fitted exponent of a heavy tailed degree distribution, `nan` below 50
            nodes or for a narrow one. A directed frame gives its in and out degrees and their moments too.
            `'assortativity'` gives the assortativity by degree, or by a node attribute, positive when linked nodes
            are alike. `'centrality'` gives a centrality of every node, or of every edge. `'communities'` detects
            communities and gives their modularity and their agreement with a reference partition. `'modularity'`
            gives the modularity of the partition `partition`. `'neighbor-degree'` gives the mean degree of the
            neighbours of nodes against their degree, rising in assortative networks. `'spectrum'` gives the adjacency
            and Laplacian eigenvalues, the algebraic connectivity with its Fiedler vector, the spectral radius, and the
            modularity matrix. It is dense, of cost `n³` per frame, and reads a directed frame as undirected for the
            Laplacian. `'summary'` gives the measures of `measures`, with the path measures on the largest component.

        orders:
            The orders `n` of the degree moments.

        attribute:
            The node attribute of the assortativity, or `None` for the degree.

        numeric:
            Whether the attribute is a number compared by correlation rather than a category compared by equality.

        measure:
            The centrality: `'degree'`, `'betweenness'`, `'closeness'`, `'eigenvector'`, `'pagerank'`, `'katz'`, or
            `'edge_betweenness'`, the betweenness of every edge, keyed by edge. The centralities are unweighted.

        alpha:
            The attenuation of the Katz centrality, below the inverse spectral radius, or `None` for 0.9 over the
            spectral radius of every frame.

        method:
            The community detection: `'louvain'`, Louvain modularity maximization, `'label_propagation'`,
            `'girvan_newman'`, the removal of the edges of highest betweenness, which suits graphs of a few hundred
            nodes, or `'leading_eigenvector'`, the splits by the leading eigenvector of the modularity matrix of
            igraph. Label propagation and the removals of Girvan-Newman ignore the weights.

        n_communities:
            The number of communities Girvan-Newman and the leading eigenvector method stop at, or `None` for the
            split of highest modularity of Girvan-Newman and as many as the leading eigenvector method finds.

        reference:
            The reference community of every node in node order, the name of a node attribute that holds it, or
            `None`.

        partition:
            The community of every node in node order, or the name of a node attribute that holds it, for
            `'modularity'`.

        weight:
            The edge attribute read as the weight of the communities, the modularity, and the spectrum, or `None` for
            an unweighted graph.

        measures:
            The measures of the summary, among `'size'`, the nodes, edges, and mean degree, `'clustering'`,
            `'transitivity'`, `'paths'`, the path length and diameter, `'components'`, and `'reciprocity'`, or `None`
            for all of them. The others stay `None`.

        samples:
            The number of source nodes from which the summary estimates the mean path length, with the diameter then
            a lower bound by the two sweep method, or `None` for the exact values.

        random_state:
            The seed of the Louvain method and of the sampled sources of the summary.

    Returns:
        result:
            The graphs and the measure of every frame.

    Raises:
        AnalysisError:
            The kind, centrality, method, or summary measure is unknown, a setting the kind, the centrality, or the
            method does not read is changed, the Katz attenuation is too large, or the kind `'modularity'` has no
            partition.

        TagError:
            The system of a trajectory has no network space.

    See Also:
        motifs: Counts the small patterns of links of a network.
        neighbors: Gives the neighbours of agents, grid cells, or network nodes.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.analysis import network_measures
        >>> halves = network_measures(nx.path_graph(4), kind='modularity', partition=[0, 0, 1, 1])
        >>> halves.modularity.round(3).tolist()
        [0.167]
    """
    _check_choice('network_measures', 'kind', kind, KINDS)
    _check_tags(network_measures, source, kind)
    _check_choice('network_measures', 'measure', measure, CENTRALITIES)
    _check_choice('network_measures', 'method', method, METHODS)
    for name in measures or ():
        _check_choice('network_measures', 'measures', name, SUMMARY_MEASURES)
    _check_settings(
        network_measures,
        'kind',
        SETTINGS[kind],
        kind=kind,
        orders=orders,
        attribute=attribute,
        numeric=numeric,
        measure=measure,
        alpha=alpha,
        method=method,
        n_communities=n_communities,
        reference=reference,
        partition=partition,
        weight=weight,
        measures=measures,
        samples=samples,
        random_state=random_state,
    )
    _check_dependent_settings(
        kind,
        measure,
        alpha=alpha,
        method=method,
        n_communities=n_communities,
        partition=partition,
    )
    graphs = _read_networks(source)
    if kind == 'summary':
        result = _summarize(graphs, SUMMARY_MEASURES if measures is None else tuple(measures), samples, random_state)
    elif kind == 'degrees':
        result = _measure_degrees(graphs, orders)
    elif kind == 'assortativity':
        assortativity = np.asarray([_assortativity(graph, attribute, numeric=numeric) for graph in graphs])
        result = NetworkMeasuresResult(graphs, assortativity=assortativity)
    elif kind == 'centrality':
        result = NetworkMeasuresResult(graphs, centrality=[_centrality(graph, measure, alpha) for graph in graphs])
    elif kind == 'neighbor-degree':
        pairs = [_neighbor_degree(graph) for graph in graphs]
        result = NetworkMeasuresResult(
            graphs,
            distinct_degrees=[degrees for degrees, _ in pairs],
            neighbor_degree=[connectivity for _, connectivity in pairs],
        )
    elif kind == 'spectrum':
        result = _measure_spectra(graphs, weight)
    elif kind == 'modularity':
        scored = [_score_partition(graph, partition, weight) for graph in graphs]
        result = NetworkMeasuresResult(
            graphs,
            labels=[labels for labels, _ in scored],
            modularity=np.asarray([modularity for _, modularity in scored]),
        )
    else:
        found = [
            _communities(
                graph,
                method,
                reference=reference,
                random_state=random_state,
                weight=weight,
                n_communities=n_communities,
            )
            for graph in graphs
        ]
        result = NetworkMeasuresResult(
            graphs,
            labels=[labels for labels, _, _ in found],
            modularity=np.asarray([modularity for _, modularity, _ in found]),
            agreement=None if reference is None else np.asarray([agreement for _, _, agreement in found]),
        )
    return result


def _check_dependent_settings(
    kind: str,
    measure: str,
    *,
    alpha: float | None,
    method: str,
    n_communities: int | None,
    partition: Sequence[object] | str | None,
) -> None:
    """Raise an error for a setting the centrality or method ignores, or no partition."""
    if alpha is not None and measure != 'katz':
        error_msg = f'network_measures of measure {measure!r} does not read alpha. Only the measure katz reads it.'
        raise AnalysisError(error_msg)
    if n_communities is not None and method not in SPLITTING_METHODS:
        error_msg = (
            f'network_measures of method {method!r} does not read n_communities. The methods '
            f'{", ".join(SPLITTING_METHODS)} read it.'
        )
        raise AnalysisError(error_msg)
    if kind == 'modularity' and partition is None:
        error_msg = (
            'network_measures of kind modularity scores a given partition. Give partition, the community of every '
            'node in node order or the name of a node attribute.'
        )
        raise AnalysisError(error_msg)


def _degrees(graph: nx.Graph) -> npt.NDArray[np.int64]:
    """Return the degree of every node, in node order."""
    return np.asarray([degree for _, degree in graph.degree()], dtype=int)


def _measure_degrees(graphs: list[nx.Graph], orders: Sequence[int]) -> NetworkMeasuresResult:
    """Return the degrees, their moments, and the exponent of every frame.

    A directed frame gives its in and out degrees and their moments too.
    """
    degrees = [_degrees(graph) for graph in graphs]
    directed = any(graph.is_directed() for graph in graphs)
    ins = [_directed_degrees(graph, 'in') for graph in graphs] if directed else None
    outs = [_directed_degrees(graph, 'out') for graph in graphs] if directed else None
    return NetworkMeasuresResult(
        graphs,
        degrees=degrees,
        moments=np.asarray([_moments(values, orders) for values in degrees]),
        exponent=np.asarray([_degree_exponent(graph) for graph in graphs]),
        in_degrees=ins,
        out_degrees=outs,
        in_moments=None if ins is None else [None if values is None else _moments(values, orders) for values in ins],
        out_moments=None if outs is None else [None if values is None else _moments(values, orders) for values in outs],
    )


def _directed_degrees(graph: nx.Graph, direction: str) -> npt.NDArray[np.int64] | None:
    """Return the in or out degree of every node of a directed graph, in node order.

    An undirected graph gives `None`.
    """
    if not graph.is_directed():
        return None
    view = graph.in_degree() if direction == 'in' else graph.out_degree()
    return np.asarray([degree for _, degree in view], dtype=int)


def _moments(values: npt.NDArray[np.int64], orders: Sequence[int]) -> npt.NDArray[np.float64]:
    """Return the moments of the degrees of every order."""
    return np.asarray([float(np.mean(values.astype(float) ** order)) for order in orders])


def _degree_exponent(graph: nx.Graph) -> float:
    """Return the fitted exponent of a heavy-tailed degree distribution.

    It is `nan` for a small or narrow one.
    """
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


def _summarize(
    graphs: list[nx.Graph],
    measures: tuple[str, ...],
    samples: int | None,
    random_state: int | None,
) -> NetworkMeasuresResult:
    """Return the chosen summary measures of every frame, with the others `None`."""
    rows = [_summary(graph, measures, samples, random_state) for graph in graphs]
    return NetworkMeasuresResult(
        graphs,
        nodes=_column(rows, 'nodes', int),
        edges=_column(rows, 'edges', int),
        mean_degree=_column(rows, 'mean_degree', float),
        clustering=_column(rows, 'clustering', float),
        transitivity=_column(rows, 'transitivity', float),
        path_length=_column(rows, 'path_length', float),
        diameter=_column(rows, 'diameter', int),
        components=(
            [components for row in rows if isinstance(components := row['components'], list)]
            if 'components' in measures
            else None
        ),
        reciprocity=_column(rows, 'reciprocity', float),
    )


def _column(rows: list[dict[str, float | int | list[int]]], name: str, dtype: type) -> npt.NDArray[Any] | None:
    """Return the values of a summary measure of every frame, or `None` without it."""
    return np.asarray([row[name] for row in rows], dtype=dtype) if rows and name in rows[0] else None


def _summary(
    graph: nx.Graph,
    measures: tuple[str, ...] = SUMMARY_MEASURES,
    samples: int | None = None,
    random_state: int | None = None,
) -> dict[str, float | int | list[int]]:
    """Return the chosen summary measures of a network.

    They are among the size, clustering, transitivity, paths, components, and reciprocity.
    """
    undirected = graph.to_undirected()
    empty = graph.number_of_nodes() == 0
    row: dict[str, float | int | list[int]] = {}
    if 'size' in measures:
        row['nodes'] = graph.number_of_nodes()
        row['edges'] = graph.number_of_edges()
        row['mean_degree'] = 0.0 if empty else float(np.mean(_degrees(undirected)))
    if 'clustering' in measures:
        row['clustering'] = 0.0 if empty else float(nx.average_clustering(undirected))
    if 'transitivity' in measures:
        row['transitivity'] = float(nx.transitivity(undirected))
    if 'paths' in measures:
        largest = undirected.subgraph(max(nx.connected_components(undirected), key=len)) if not empty else undirected
        row['path_length'], row['diameter'] = _paths(largest, samples, random_state)
    if 'components' in measures:
        row['components'] = sorted((len(component) for component in nx.connected_components(undirected)), reverse=True)
    if 'reciprocity' in measures:
        directed = graph.is_directed() and graph.number_of_edges() > 0
        row['reciprocity'] = float(nx.overall_reciprocity(graph)) if directed else float('nan')
    return row


def _paths(largest: nx.Graph, samples: int | None, random_state: int | None) -> tuple[float, int]:
    """Return the mean shortest path length and the diameter of a connected graph.

    With fewer samples than nodes, the mean is over the paths from that many seeded sources, and the diameter is the
    two sweep lower bound.
    """
    size = largest.number_of_nodes()
    if size <= 1:
        return 0.0, 0
    if samples is None or samples >= size:
        return float(nx.average_shortest_path_length(largest)), int(nx.diameter(largest))
    nodes = list(largest)
    sources = np.random.default_rng(random_state).choice(size, size=samples, replace=False)
    total = sum(sum(nx.single_source_shortest_path_length(largest, nodes[index]).values()) for index in sources)
    return float(total / (samples * (size - 1))), int(nx.approximation.diameter(largest, seed=random_state))


def _assortativity(graph: nx.Graph, attribute: str | None, *, numeric: bool) -> float:
    """Return the assortativity of a network.

    It is by degree, or by a categorical or numeric node attribute.
    """
    if attribute is None:
        return float(nx.degree_assortativity_coefficient(graph))
    if numeric:
        return float(nx.numeric_assortativity_coefficient(graph, attribute))
    return float(nx.attribute_assortativity_coefficient(graph, attribute))


def _centrality(graph: nx.Graph, measure: str, alpha: float | None) -> dict[object, float]:
    """Return a centrality of every node, or the betweenness of every edge.

    The centrality is the degree, betweenness, closeness, eigenvector, PageRank, or Katz centrality.
    """
    if measure == 'katz':
        return _katz(graph, alpha)
    functions = {
        'degree': nx.degree_centrality,
        'betweenness': nx.betweenness_centrality,
        'closeness': nx.closeness_centrality,
        'eigenvector': nx.eigenvector_centrality_numpy,
        'pagerank': nx.pagerank,
        'edge_betweenness': nx.edge_betweenness_centrality,
    }
    return dict(functions[measure](graph))


def _katz(graph: nx.Graph, alpha: float | None) -> dict[object, float]:
    """Return the Katz centrality of every node.

    The attenuation is `alpha`, or 0.9 over the spectral radius, and it stays below the inverse spectral radius.
    """
    radius = float(np.max(np.abs(nx.adjacency_spectrum(graph, weight=None)))) if graph.number_of_nodes() else 0.0
    if alpha is None:
        alpha = KATZ_SHARE / radius if radius > 0 else 0.0
    elif alpha * radius >= 1:
        error_msg = (
            f'network_measures has alpha={alpha!r}, at or above the inverse spectral radius {1 / radius:.6g} of a '
            'frame, where the Katz series diverges. Give a smaller alpha, or None for 0.9 of it.'
        )
        raise AnalysisError(error_msg)
    return dict(nx.katz_centrality_numpy(graph, alpha=alpha, beta=1.0))


def _neighbor_degree(graph: nx.Graph) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Return the distinct degrees and the mean degree of their nodes' neighbours.

    The mean is over the neighbours of the nodes of every degree.
    """
    connectivity = nx.average_degree_connectivity(graph)
    ordered = sorted(connectivity)
    return np.asarray(ordered, dtype=np.int64), np.asarray(
        [connectivity[degree] for degree in ordered],
        dtype=float,
    )


def _measure_spectra(graphs: list[nx.Graph], weight: str | None) -> NetworkMeasuresResult:
    """Return the spectra, the connectivity, and the modularity matrix of every frame."""
    adjacency = [_adjacency_spectrum(graph, weight) for graph in graphs]
    undirected = [graph.to_undirected() for graph in graphs]
    connectivity = [_algebraic_connectivity(graph, weight) for graph in undirected]
    return NetworkMeasuresResult(
        graphs,
        adjacency_spectrum=adjacency,
        laplacian_spectrum=[np.sort(nx.laplacian_spectrum(graph, weight=weight)) for graph in undirected],
        normalized_laplacian_spectrum=[
            np.sort(nx.normalized_laplacian_spectrum(graph, weight=weight)) for graph in undirected
        ],
        algebraic_connectivity=np.asarray([value for value, _ in connectivity]),
        fiedler_vector=[vector for _, vector in connectivity],
        spectral_radius=np.asarray([float(np.max(np.abs(values))) if values.size else 0.0 for values in adjacency]),
        modularity_matrix=[_modularity_matrix(graph, weight) for graph in graphs],
    )


def _adjacency_spectrum(graph: nx.Graph, weight: str | None) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """Return the adjacency eigenvalues.

    They are real and ascending for an undirected graph, and complex and by descending modulus for a directed one.
    """
    values = nx.adjacency_spectrum(graph, weight=weight)
    if not graph.is_directed():
        return np.sort(values.real)
    return values[np.argsort(-np.abs(values), kind='stable')]


def _algebraic_connectivity(graph: nx.Graph, weight: str | None) -> tuple[float, npt.NDArray[np.float64]]:
    """Return the algebraic connectivity and the Fiedler vector of an undirected graph.

    A disconnected graph gives 0 and a vector of `nan`, and a graph below two nodes 0 and an empty vector.
    """
    size = graph.number_of_nodes()
    if size <= 1:
        return 0.0, np.empty(0)
    if not nx.is_connected(graph):
        return 0.0, np.full(size, np.nan)
    value = float(nx.algebraic_connectivity(graph, weight=weight, seed=0))
    return value, np.asarray(nx.fiedler_vector(graph, weight=weight, seed=0), dtype=float)


def _modularity_matrix(graph: nx.Graph, weight: str | None) -> npt.NDArray[np.float64]:
    """Return the modularity matrix of a graph in node order.

    A directed graph gives the directed modularity matrix.
    """
    if graph.is_directed():
        return np.asarray(nx.directed_modularity_matrix(graph, weight=weight), dtype=float)
    return np.asarray(nx.modularity_matrix(graph, weight=weight), dtype=float)


def _score_partition(
    graph: nx.Graph,
    partition: Sequence[object] | str | None,
    weight: str | None,
) -> tuple[npt.NDArray[np.int64], float]:
    """Return the labels of a given partition and its modularity.

    The labels are recoded from 0 in order of first appearance.
    """
    given = [graph.nodes[node][partition] for node in graph.nodes()] if isinstance(partition, str) else partition
    codes: dict[object, int] = {}
    labels = np.asarray([codes.setdefault(label, len(codes)) for label in given or ()], dtype=np.int64)
    return labels, float(nx.community.modularity(graph, _groups(graph, labels), weight=weight))


def _groups(graph: nx.Graph, labels: Sequence[int] | npt.NDArray[np.int64]) -> list[set[object]]:
    """Return the nodes of every community from the labels in node order."""
    groups: dict[int, set[object]] = {}
    for node, label in zip(graph.nodes(), labels, strict=True):
        groups.setdefault(int(label), set()).add(node)
    return list(groups.values())


def _communities(
    graph: nx.Graph,
    method: str,
    *,
    reference: Sequence[object] | str | None,
    random_state: int | None,
    weight: str | None,
    n_communities: int | None,
) -> tuple[npt.NDArray[np.int64], float, float | None]:
    """Return the community of every node, the modularity, and the agreement.

    The agreement is with a reference partition.
    """
    if method == 'louvain':
        partition = nx.community.louvain_communities(graph, weight=weight, seed=random_state)
    elif method == 'label_propagation':
        partition = list(nx.community.label_propagation_communities(graph.to_undirected()))
    elif method == 'girvan_newman':
        partition = _girvan_newman(graph, weight, n_communities)
    else:
        partition = _leading_eigenvector(graph, weight, n_communities)
    membership = {node: index for index, members in enumerate(partition) for node in members}
    labels = np.asarray([membership[node] for node in graph.nodes()], dtype=np.int64)
    modularity = float(nx.community.modularity(graph, partition, weight=weight))
    if reference is None:
        return labels, modularity, None
    truth = [graph.nodes[node][reference] for node in graph.nodes()] if isinstance(reference, str) else reference
    return labels, modularity, float(normalized_mutual_info_score(list(truth), labels))


def _girvan_newman(graph: nx.Graph, weight: str | None, n_communities: int | None) -> list[set[object]]:
    """Return the communities of the Girvan-Newman hierarchy.

    It is the first split of `n_communities` communities, or the split of highest modularity without it. The edges are
    removed by unweighted betweenness, and the weight enters only the modularity.
    """
    undirected = graph.to_undirected()
    found = [set(component) for component in nx.connected_components(undirected)]
    splits = nx.community.girvan_newman(undirected)
    if n_communities is None:
        best = max(
            [found, *([set(community) for community in split] for split in splits)],
            key=lambda split: nx.community.modularity(graph, split, weight=weight),
        )
        return list(best)
    for split in splits:
        if len(found) >= n_communities:
            break
        found = [set(community) for community in split]
    return found


def _leading_eigenvector(graph: nx.Graph, weight: str | None, n_communities: int | None) -> list[set[object]]:
    """Return the communities of the leading eigenvector method of igraph.

    It splits the undirected graph.
    """
    undirected = graph.to_undirected()
    weights = None if weight is None else [float(data.get(weight, 1.0)) for _, _, data in undirected.edges(data=True)]
    converted = load_module('igraph').Graph.from_networkx(undirected)
    membership = converted.community_leading_eigenvector(clusters=n_communities, weights=weights).membership
    return _groups(undirected, membership)
