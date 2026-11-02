"""Measure networks: degrees, mixing, centralities, communities, neighbour degrees, and summary measures."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import networkx as nx
import numpy as np
import numpy.typing as npt
from sklearn.metrics import normalized_mutual_info_score

from ..types import DataKind
from ._base import BaseAnalysis
from ._fit import FitAnalysis
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


class NetworkAnalysis(BaseAnalysis):
    """Measures of the network of every frame of a trajectory, or of a graph, of one kind.

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

    Attributes:
        graphs_:
            The graph of every frame.

        degrees_:
            The degree of every node of every frame.

        moments_:
            The degree moments, frames by orders.

        exponent_:
            The fitted degree exponent of every frame.

        assortativity_:
            The assortativity of every frame.

        centrality_:
            A mapping from node to its centrality for every frame.

        labels_:
            The community of every node of every frame.

        modularity_:
            The modularity of every frame.

        agreement_:
            The normalized mutual information with the reference of every frame, or `None` without a reference.

        distinct_degrees_:
            The distinct degrees of every frame.

        neighbor_degree_:
            The mean neighbour degree of the nodes of every distinct degree of every frame.

        nodes_, edges_, mean_degree_, clustering_, path_length_, diameter_:
            The summary measures of every frame.

        components_:
            The sizes of the connected components of every frame, largest first.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.analysis import NetworkAnalysis
        >>> summary = NetworkAnalysis().check().prepare(nx.cycle_graph(10)).run()
        >>> summary.mean_degree_.tolist(), summary.clustering_.tolist(), summary.diameter_.tolist()
        ([2.0], [0.0], [5])
        >>> NetworkAnalysis('degrees').check().prepare(nx.star_graph(3)).run().moments_.tolist()
        [[1.5, 3.0]]
        >>> centrality = NetworkAnalysis('centrality', measure='betweenness').check().prepare(nx.star_graph(3)).run()
        >>> centrality.centrality_[0][0]
        1.0
    """

    def __init__(
        self,
        kind: str = 'summary',
        *,
        orders: Sequence[int] = (1, 2),
        attribute: str | None = None,
        numeric: bool = False,
        measure: str = 'degree',
        method: str = 'louvain',
        reference: Sequence[object] | str | None = None,
        random_state: int | None = None,
    ) -> None:
        self.kind = kind
        self.orders = orders
        self.attribute = attribute
        self.numeric = numeric
        self.measure = measure
        self.method = method
        self.reference = reference
        self.random_state = random_state

    def _check(self) -> None:
        """Validate the kind, the centrality, the method, and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_choice('measure', self.measure, CENTRALITIES)
        self._check_choice('method', self.method, METHODS)
        self._check_settings('kind', SETTINGS[self.kind])

    def _get_data_kind(self) -> DataKind:
        """Return the network kind."""
        return DataKind.NETWORK

    def _prepare(self, source: Source) -> None:
        """Read the graph of every frame of a trajectory, or a graph."""
        self.graphs_ = self._read_networks(source)

    def _run_summary(self) -> None:
        """Compute the summary measures of every frame."""
        rows = [self._summary(graph) for graph in self.graphs_]
        self.nodes_ = np.asarray([row['nodes'] for row in rows], dtype=int)
        self.edges_ = np.asarray([row['edges'] for row in rows], dtype=int)
        self.mean_degree_ = np.asarray([row['mean_degree'] for row in rows], dtype=float)
        self.clustering_ = np.asarray([row['clustering'] for row in rows], dtype=float)
        self.path_length_ = np.asarray([row['path_length'] for row in rows], dtype=float)
        self.diameter_ = np.asarray([row['diameter'] for row in rows], dtype=int)
        self.components_ = [row['components'] for row in rows]

    def _run(self, inputs: object = None) -> None:
        """Compute the measure of every frame."""
        graphs = self.graphs_
        if self.kind == 'summary':
            self._run_summary()
            return
        if self.kind == 'degrees':
            self.degrees_ = [self._degrees(graph) for graph in graphs]
            self.moments_ = np.asarray(
                [[float(np.mean(values.astype(float) ** order)) for order in self.orders] for values in self.degrees_],
            )
            self.exponent_ = np.asarray([self._degree_exponent(graph) for graph in graphs])
        elif self.kind == 'assortativity':
            self.assortativity_ = np.asarray(
                [self._assortativity(graph, self.attribute, numeric=self.numeric) for graph in graphs],
            )
        elif self.kind == 'centrality':
            self.centrality_ = [self._centrality(graph, self.measure) for graph in graphs]
        elif self.kind == 'neighbor-degree':
            pairs = [self._neighbor_degree(graph) for graph in graphs]
            self.distinct_degrees_ = [degrees for degrees, _ in pairs]
            self.neighbor_degree_ = [connectivity for _, connectivity in pairs]
        else:
            found = [self._communities(graph, self.method, self.reference, self.random_state) for graph in graphs]
            self.labels_ = [labels for labels, _, _ in found]
            self.modularity_ = np.asarray([modularity for _, modularity, _ in found])
            self.agreement_ = None if self.reference is None else np.asarray([agreement for _, _, agreement in found])

    @staticmethod
    def _degrees(graph: nx.Graph) -> npt.NDArray[np.int64]:
        """Return the degree of every node, in node order."""
        return np.asarray([degree for _, degree in graph.degree()], dtype=int)

    @staticmethod
    def _degree_exponent(graph: nx.Graph) -> float:
        """Return the fitted exponent of a heavy-tailed degree distribution, or `nan` for a small or narrow one."""
        values = NetworkAnalysis._degrees(graph)
        if values.size < MIN_FIT_NODES or values.max() < HEAVY_TAIL * max(values.mean(), 1.0):
            return float('nan')
        fit = FitAnalysis(discrete=True, bootstrap=0, candidates=FIT_CANDIDATES, compare_with=())
        return fit.check().prepare(values[values > 0]).run().exponent_

    @staticmethod
    def _summary(graph: nx.Graph) -> dict[str, float | int | list[int]]:
        """Return the size, mean degree, clustering, path length, diameter, and components of a network."""
        undirected = graph.to_undirected()
        components = sorted((len(component) for component in nx.connected_components(undirected)), reverse=True)
        largest = undirected.subgraph(max(nx.connected_components(undirected), key=len)) if components else undirected
        return {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'mean_degree': float(np.mean(NetworkAnalysis._degrees(undirected))) if graph.number_of_nodes() else 0.0,
            'clustering': float(nx.average_clustering(undirected)) if graph.number_of_nodes() else 0.0,
            'path_length': float(nx.average_shortest_path_length(largest)) if largest.number_of_nodes() > 1 else 0.0,
            'diameter': int(nx.diameter(largest)) if largest.number_of_nodes() > 1 else 0,
            'components': components,
        }

    @staticmethod
    def _assortativity(graph: nx.Graph, attribute: str | None, *, numeric: bool) -> float:
        """Return the assortativity of a network by degree, or by a categorical or numeric node attribute."""
        if attribute is None:
            return float(nx.degree_assortativity_coefficient(graph))
        if numeric:
            return float(nx.numeric_assortativity_coefficient(graph, attribute))
        return float(nx.attribute_assortativity_coefficient(graph, attribute))

    @staticmethod
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

    @staticmethod
    def _neighbor_degree(graph: nx.Graph) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Return the distinct degrees and the mean degree of the neighbours of the nodes of every degree."""
        connectivity = nx.average_degree_connectivity(graph)
        ordered = sorted(connectivity)
        return np.asarray(ordered, dtype=np.int64), np.asarray(
            [connectivity[degree] for degree in ordered],
            dtype=float,
        )

    @staticmethod
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
