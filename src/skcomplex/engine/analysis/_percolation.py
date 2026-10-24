"""Measure clusters and percolation: cluster sizes of grids and graphs, the largest cluster, and spanning."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt

from ._kinds import DataKind, analysis
from ._spatial import label_clusters

type Array = npt.NDArray[np.float64]


@analysis(DataKind.FIELD)
def cluster_sizes(
    data: npt.ArrayLike | nx.Graph,
    *,
    periodic: bool = False,
    diagonal: bool = False,
) -> npt.NDArray[np.int64]:
    """Return the sizes of the clusters of occupied cells of a grid, or of the connected components of a graph.

    Args:
        data:
            A boolean grid of any dimension, or a NetworkX graph.

        periodic:
            Whether clusters of a grid wrap around its edges.

        diagonal:
            Whether cells that share only a corner are connected.

    Returns:
        sizes:
            The size of every cluster, largest first.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.analysis import cluster_sizes
        >>> cluster_sizes([[1, 1, 0], [0, 0, 1]]).tolist(), cluster_sizes(nx.path_graph(3)).tolist()
        ([2, 1], [3])
    """
    if isinstance(data, nx.Graph):
        sizes = [len(component) for component in nx.connected_components(data.to_undirected(as_view=True))]
        return np.sort(np.asarray(sizes, dtype=np.int64))[::-1]
    return np.sort(label_clusters(data, periodic=periodic, diagonal=diagonal).sizes)[::-1]


@dataclass(frozen=True)
class Percolation:
    """Percolation observables of one grid or graph.

    Args:
        sizes:
            The size of every cluster, largest first.

        strength:
            The share of all sites or nodes in the largest cluster.

        mean_size:
            The mean size of the cluster of an occupied site outside the largest, `Σ' s² / Σ' s`.

        spanning:
            Whether a cluster of a grid joins the two opposite edges of its first axis, always false for graphs.
    """

    sizes: npt.NDArray[np.int64]
    strength: float
    mean_size: float
    spanning: bool


@analysis(DataKind.FIELD)
def percolation(data: npt.ArrayLike | nx.Graph, *, diagonal: bool = False) -> Percolation:
    """Return the cluster sizes, the strength of the largest cluster, the mean cluster size, and spanning.

    Args:
        data:
            A boolean grid of any dimension, or a NetworkX graph.

        diagonal:
            Whether cells that share only a corner are connected.

    Returns:
        percolation:
            The percolation observables.

    Examples:
        >>> from skcomplex.core.analysis import percolation
        >>> result = percolation([[1, 1, 0], [1, 0, 1]])
        >>> result.strength, result.mean_size, result.spanning
        (0.5, 1.0, True)
    """
    if isinstance(data, nx.Graph):
        sizes = cluster_sizes(data)
        total, spanning = data.number_of_nodes(), False
    else:
        grid = np.asarray(data, dtype=bool)
        clusters = label_clusters(grid, diagonal=diagonal)
        sizes = np.sort(clusters.sizes)[::-1]
        total, spanning = grid.size, clusters.spans(0)
    rest = sizes[1:].astype(float)
    return Percolation(
        sizes,
        float(sizes[0] / total) if sizes.size else 0.0,
        float(np.sum(rest**2) / rest.sum()) if rest.size else 0.0,
        spanning,
    )


@analysis(DataKind.FIELD)
def spanning_probability(grids: Sequence[npt.ArrayLike], *, diagonal: bool = False) -> float:
    """Return the share of grids of an ensemble with a cluster joining the opposite edges of their first axis.

    Args:
        grids:
            The boolean grids.

        diagonal:
            Whether cells that share only a corner are connected.

    Returns:
        probability:
            The spanning probability.

    Examples:
        >>> from skcomplex.core.analysis import spanning_probability
        >>> spanning_probability([[[1], [1]], [[1], [0]]])
        0.5
    """
    return float(np.mean([label_clusters(grid, diagonal=diagonal).spans(0) for grid in grids]))
