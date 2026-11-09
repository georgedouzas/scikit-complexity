"""Count network motifs of three and four nodes and score them.

The scores are against degree-preserving null ensembles.

`SETTINGS` holds the settings every kind reads, besides `kind`. `ISOCLASSES` holds the number of isomorphism classes of
igraph of every size, undirected and directed.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import itertools
from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt

from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import DataKind
from ._base import _check_choice, _check_settings, _check_tags, _read_networks
from ._utils import load_module
from .types import Source

TRIAD_SIZE = 3
KINDS = ('counts', 'z-scores')
NULLS = ('switch', 'configuration')
SIZES = (3, 4)
SETTINGS = {
    'counts': {'size'},
    'z-scores': {'size', 'null', 'samples', 'random_state'},
}
SWAPS_PER_EDGE = 3
UNDIRECTED_THREE = {2: 'path', 3: 'triangle'}
UNDIRECTED_FOUR = {
    (3, (1, 1, 1, 3)): 'star',
    (3, (1, 1, 2, 2)): 'path',
    (4, (2, 2, 2, 2)): 'cycle',
    (4, (1, 2, 2, 3)): 'tailed_triangle',
    (5, (2, 2, 3, 3)): 'diamond',
    (6, (3, 3, 3, 3)): 'clique',
}
ISOCLASSES = {(3, False): 4, (3, True): 16, (4, False): 11, (4, True): 218}
PERMUTATIONS = tuple(itertools.permutations(range(4)))
TAGS = AnalysisTags(DataKind.NETWORK, space=frozenset({'network'}))


@dataclass(frozen=True)
class MotifsResult:
    """Motifs of three or four nodes of a graph or of the network of every frame.

    The networks are those of a trajectory.

    Args:
        graphs:
            The graph of every frame.

        counts:
            The count of every motif of every frame.

        null_mean:
            The mean count of every motif over the null ensemble of every frame, for `'z-scores'`.

        null_std:
            The standard deviation of every count over the null ensemble of every frame, for `'z-scores'`.

        z_scores:
            The z-score of every motif of every frame, for `'z-scores'`.
    """

    graphs: list[nx.Graph]
    counts: list[dict[str, int]]
    null_mean: list[dict[str, float]] | None = None
    null_std: list[dict[str, float]] | None = None
    z_scores: list[dict[str, float]] | None = None


def motifs(
    source: Source,
    *,
    kind: str = 'counts',
    size: int = 3,
    null: str = 'switch',
    samples: int = 100,
    random_state: int | None = None,
) -> MotifsResult:
    """Return the motifs of three or four nodes of a graph or of every frame's network.

    The networks are those of a trajectory.

    The kinds count or score motifs:

    - `'counts'`: the number of connected induced subgraphs of every kind. Directed triads carry the names of the
      triadic census, such as `'030T'` for the feed-forward loop. Undirected motifs are named by their shape, and
      directed motifs of four nodes by their smallest adjacency code.
    - `'z-scores'`: the counts against random graphs that keep the degrees, `(count - null_mean) / null_std` of every
      motif, `nan` where the null count never varies.

    Args:
        source:
            The trajectory, a graph, or a network snapshot.

        kind:
            `'counts'` or `'z-scores'`.

        size:
            The number of nodes of the motifs, 3 or 4.

        null:
            `'switch'` for degree-preserving edge switches, or `'configuration'` for the configuration system with
            self-loops and multiple edges removed.

        samples:
            The number of random graphs.

        random_state:
            The seed of the ensemble.

    Returns:
        result:
            The graphs and the counts or scores of their motifs.

    Raises:
        AnalysisError:
            The kind, size, or null ensemble is unknown, a setting the kind does not read is changed, or the number of
            samples is below 1.

        TagError:
            The system of a trajectory has no network space.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.analysis import motifs
        >>> motifs(nx.complete_graph(4)).counts
        [{'path': 0, 'triangle': 4}]
        >>> motifs(nx.complete_graph(4), size=4).counts
        [{'clique': 1}]
        >>> graph = nx.gnm_random_graph(30, 60, seed=0)
        >>> sorted(motifs(graph, kind='z-scores', samples=10, random_state=0).z_scores[0])
        ['path', 'triangle']
    """
    _check_choice('motifs', 'kind', kind, KINDS)
    _check_tags(motifs, source, kind)
    _check_choice('motifs', 'size', size, SIZES)
    _check_choice('motifs', 'null', null, NULLS)
    _check_settings(
        motifs,
        'kind',
        SETTINGS[kind],
        kind=kind,
        size=size,
        null=null,
        samples=samples,
        random_state=random_state,
    )
    if samples < 1:
        error_msg = f'motifs has samples={samples!r}. Use 1 or more.'
        raise AnalysisError(error_msg)
    graphs = _read_networks(source)
    if kind == 'counts':
        return MotifsResult(graphs, [_motif_counts(graph, size) for graph in graphs])
    scores = [_motif_scores(graph, size, null, samples, random_state) for graph in graphs]
    return MotifsResult(
        graphs,
        [counts for counts, _, _, _ in scores],
        null_mean=[means for _, means, _, _ in scores],
        null_std=[deviations for _, _, deviations, _ in scores],
        z_scores=[values for _, _, _, values in scores],
    )


def _directed_code(adjacency: npt.NDArray[np.int64]) -> str:
    """Return a name of the isomorphism class of a directed motif of four nodes.

    The name is its smallest adjacency code.
    """
    codes = []
    for permutation in PERMUTATIONS:
        permuted = adjacency[np.ix_(permutation, permutation)]
        codes.append(int(''.join(map(str, permuted[~np.eye(4, dtype=bool)])), 2))
    return f'directed_{min(codes)}'


def _motif_name(adjacency: npt.NDArray[np.int64], directed: bool) -> str:
    """Return the name of the connected motif of an adjacency matrix.

    The matrix has three or four nodes.
    """
    if directed and len(adjacency) == TRIAD_SIZE:
        census = nx.triadic_census(nx.from_numpy_array(adjacency, create_using=nx.DiGraph))
        return next(name for name, count in census.items() if count)
    if directed:
        return _directed_code(adjacency)
    edges = int(adjacency.sum()) // 2
    if len(adjacency) == TRIAD_SIZE:
        return UNDIRECTED_THREE[edges]
    return UNDIRECTED_FOUR[(edges, tuple(sorted(adjacency.sum(axis=0).tolist())))]


@functools.cache
def _motif_names(size: int, directed: bool) -> tuple[str | None, ...]:
    """Return the name of every isomorphism class of igraph of a size.

    The classes not connected get `None`.
    """
    graph = load_module('igraph').Graph
    names: list[str | None] = []
    for isoclass in range(ISOCLASSES[(size, directed)]):
        motif = graph.Isoclass(size, isoclass, directed=directed)
        adjacency = np.asarray(motif.get_adjacency().data, dtype=np.int64)
        names.append(_motif_name(adjacency, directed) if motif.is_connected(mode='weak') else None)
    return tuple(names)


def _motif_counts(graph: nx.Graph, size: int) -> dict[str, int]:
    """Return the number of connected induced subgraphs of every kind.

    The kinds have three or four nodes. Every kind of three nodes is counted, and only the kinds of four nodes that
    occur.
    """
    names = _motif_names(size, graph.is_directed())
    counts = load_module('igraph').Graph.from_networkx(graph).motifs_randesu(size=size)
    return {
        name: int(count)
        for name, count in zip(names, counts, strict=True)
        if name is not None and (size == TRIAD_SIZE or count)
    }


def _null_graph(graph: nx.Graph, null: str, rng: np.random.Generator) -> nx.Graph:
    """Return a random graph with the degrees of a graph.

    It is drawn by edge switches or by the configuration system.
    """
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


def _motif_scores(
    graph: nx.Graph,
    size: int,
    null: str,
    samples: int,
    random_state: int | None,
) -> tuple[dict[str, int], dict[str, float], dict[str, float], dict[str, float]]:
    """Return the counts, null means, deviations, and z-scores of a graph's motifs."""
    counts = _motif_counts(graph, size)
    rng = np.random.default_rng(random_state)
    ensemble = [_motif_counts(_null_graph(graph, null, rng), size) for _ in range(samples)]
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
