"""Count network motifs of three and four nodes and score them against degree-preserving null ensembles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections import Counter
from typing import Self

import networkx as nx
import numpy as np

from ..errors import AnalysisError
from ..types import DataKind
from ._base import BaseAnalysis
from .types import Source

TRIAD_SIZE = 3

KINDS = ('counts', 'z-scores')
NULLS = ('switch', 'configuration')
SIZES = (3, 4)
SETTINGS = {
    'counts': {'size'},
    'z-scores': {'size', 'null', 'samples', 'random_state'},
}
"""Settings every kind reads, besides `kind`."""
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


class MotifAnalysis(BaseAnalysis):
    """Motifs of three or four nodes of the network of every frame of a trajectory, or of a graph.

    The kinds count or score motifs:

    - `'counts'`: the number of connected induced subgraphs of every kind. Directed triads carry the names of the
      triadic census, such as `'030T'` for the feed-forward loop. Undirected motifs are named by their shape, and
      directed motifs of four nodes by their smallest adjacency code.
    - `'z-scores'`: the counts against random graphs that keep the degrees, `(count - null_mean) / null_std` of every
      motif, `nan` where the null count never varies.

    Args:
        kind:
            `'counts'` or `'z-scores'`.

        size:
            The number of nodes of the motifs, 3 or 4.

        null:
            `'switch'` for degree-preserving edge switches, or `'configuration'` for the configuration model with
            self-loops and multiple edges removed.

        samples:
            The number of random graphs.

        random_state:
            The seed of the ensemble.

    Attributes:
        graphs_:
            The graph of every frame.

        counts_:
            The count of every motif of every frame.

        null_mean_:
            The mean count of every motif over the null ensemble of every frame.

        null_std_:
            The standard deviation of every count over the null ensemble of every frame.

        z_scores_:
            The z-score of every motif of every frame.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.analysis import MotifAnalysis
        >>> MotifAnalysis().check().prepare(nx.complete_graph(4)).run().counts_
        [{'path': 0, 'triangle': 4}]
        >>> MotifAnalysis(size=4).check().prepare(nx.complete_graph(4)).run().counts_
        [{'clique': 1}]
        >>> graph = nx.gnm_random_graph(30, 60, seed=0)
        >>> sorted(MotifAnalysis('z-scores', samples=10, random_state=0).check().prepare(graph).run().z_scores_[0])
        ['path', 'triangle']
    """

    def __init__(
        self,
        kind: str = 'counts',
        *,
        size: int = 3,
        null: str = 'switch',
        samples: int = 100,
        random_state: int | None = None,
    ) -> None:
        self.kind = kind
        self.size = size
        self.null = null
        self.samples = samples
        self.random_state = random_state

    def _check(self) -> Self:
        """Validate the kind, the size, the null ensemble, the samples, and that the kind reads every setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_choice('size', self.size, SIZES)
        self._check_choice('null', self.null, NULLS)
        self._check_settings('kind', SETTINGS[self.kind])
        if self.samples < 1:
            error_msg = f'{type(self).__name__} has samples={self.samples!r}. Use 1 or more.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the network kind."""
        return DataKind.NETWORK

    def _prepare(self, source: Source) -> Self:
        """Read the graph of every frame of a trajectory, or a graph."""
        self.graphs_ = self._read_networks(source)
        return self

    def _run(self, inputs: object = None) -> Self:
        """Count or score the motifs of every frame."""
        if self.kind == 'counts':
            self.counts_ = [self._motif_counts(graph, self.size) for graph in self.graphs_]
            return self
        scores = [
            self._motif_scores(graph, self.size, self.null, self.samples, self.random_state) for graph in self.graphs_
        ]
        self.counts_ = [counts for counts, _, _, _ in scores]
        self.null_mean_ = [means for _, means, _, _ in scores]
        self.null_std_ = [deviations for _, _, deviations, _ in scores]
        self.z_scores_ = [values for _, _, _, values in scores]
        return self

    @staticmethod
    def _extend(
        graph: nx.Graph,
        subgraph: set[object],
        extension: set[object],
        root: object,
        size: int,
    ) -> list[frozenset]:
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
            found.extend(MotifAnalysis._extend(graph, subgraph | {node}, candidates | exclusive, root, size))
        return found

    @staticmethod
    def _connected_sets(graph: nx.Graph, size: int) -> list[frozenset]:
        """Return every connected set of nodes of a size, once."""
        undirected = graph.to_undirected(as_view=True)
        order = {node: index for index, node in enumerate(undirected.nodes())}
        found = []
        for root in undirected.nodes():
            extension = {other for other in undirected.neighbors(root) if order[other] > order[root]}
            found.extend(MotifAnalysis._extend(undirected, {root}, extension, root, size))
        return found

    @staticmethod
    def _directed_code(graph: nx.DiGraph, nodes: tuple[object, ...]) -> str:
        """Return a name of the isomorphism class of a directed subgraph of four nodes, its smallest adjacency code."""
        adjacency = nx.to_numpy_array(graph, nodelist=list(nodes), dtype=int)
        codes = []
        for permutation in PERMUTATIONS:
            permuted = adjacency[np.ix_(permutation, permutation)]
            codes.append(int(''.join(map(str, permuted[~np.eye(4, dtype=bool)])), 2))
        return f'directed_{min(codes)}'

    @staticmethod
    def _four_node_name(graph: nx.Graph, nodes: frozenset) -> str:
        """Return the name of the motif a connected set of four nodes induces."""
        subgraph = graph.subgraph(nodes)
        if graph.is_directed():
            return MotifAnalysis._directed_code(subgraph, tuple(nodes))
        key = (subgraph.number_of_edges(), tuple(sorted(degree for _, degree in subgraph.degree())))
        return UNDIRECTED_FOUR[key]

    @staticmethod
    def _three_node_counts(graph: nx.Graph) -> dict[str, int]:
        """Return the triadic census of a directed graph, or the open and closed triads of an undirected one."""
        if graph.is_directed():
            census = nx.triadic_census(graph)
            return {name: int(count) for name, count in census.items() if name not in {'003', '012', '102'}}
        triangles = sum(nx.triangles(graph).values()) // 3
        wedges = sum(degree * (degree - 1) // 2 for _, degree in graph.degree())
        return {'path': int(wedges - 3 * triangles), 'triangle': int(triangles)}

    @staticmethod
    def _motif_counts(graph: nx.Graph, size: int) -> dict[str, int]:
        """Return the number of connected induced subgraphs of every kind of three or four nodes."""
        if size == TRIAD_SIZE:
            return MotifAnalysis._three_node_counts(graph)
        return dict(
            Counter(
                MotifAnalysis._four_node_name(graph, nodes) for nodes in MotifAnalysis._connected_sets(graph, size)
            ),
        )

    @staticmethod
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

    @staticmethod
    def _motif_scores(
        graph: nx.Graph,
        size: int,
        null: str,
        samples: int,
        random_state: int | None,
    ) -> tuple[dict[str, int], dict[str, float], dict[str, float], dict[str, float]]:
        """Return the counts, the null means and deviations, and the z-scores of the motifs of a graph."""
        counts = MotifAnalysis._motif_counts(graph, size)
        rng = np.random.default_rng(random_state)
        ensemble = [
            MotifAnalysis._motif_counts(MotifAnalysis._null_graph(graph, null, rng), size) for _ in range(samples)
        ]
        names = sorted(set(counts).union(*ensemble))
        table = np.asarray([[sample.get(name, 0) for name in names] for sample in ensemble], dtype=float)
        means, deviations = table.mean(axis=0), table.std(axis=0)
        observed = np.asarray([counts.get(name, 0) for name in names], dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = np.where(deviations > 0, (observed - means) / deviations, np.nan)
        return (
            {name: int(counts.get(name, 0)) for name in names},
            dict(zip(names, means.tolist(), strict=True)),
            dict(zip(names, deviations.tolist(), strict=True)),
            dict(zip(names, scores.tolist(), strict=True)),
        )
