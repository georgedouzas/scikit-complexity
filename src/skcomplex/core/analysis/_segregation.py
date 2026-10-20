"""Measure the segregation of categorical states on grids and networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import networkx as nx
import numpy as np
import numpy.typing as npt

from ..base import AnalysisError
from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]


def _shifts(ndim: int, *, diagonal: bool) -> list[tuple[int, ...]]:
    """Return the offsets of the neighbours of a cell, with or without corners."""
    offsets = np.stack(np.meshgrid(*([[-1, 0, 1]] * ndim), indexing='ij'), axis=-1).reshape(-1, ndim)
    steps = np.abs(offsets).sum(axis=1)
    keep = (steps > 0) if diagonal else (steps == 1)
    return [tuple(int(item) for item in offset) for offset in offsets[keep]]


def _grid_pairs(grid: npt.NDArray[np.float64], *, diagonal: bool) -> tuple[Array, Array]:
    """Return the states of every cell paired with every neighbour on a periodic grid."""
    pairs = [
        (grid, np.roll(grid, shift, axis=tuple(range(grid.ndim)))) for shift in _shifts(grid.ndim, diagonal=diagonal)
    ]
    return np.concatenate([first.ravel() for first, _ in pairs]), np.concatenate(
        [second.ravel() for _, second in pairs],
    )


def _graph_pairs(graph: nx.Graph, attribute: str) -> tuple[Array, Array]:
    """Return the states of both ends of every edge, both ways."""
    states = nx.get_node_attributes(graph, attribute)
    if len(states) < graph.number_of_nodes():
        error_msg = f'Some nodes have no attribute {attribute!r}. Set it on every node.'
        raise AnalysisError(error_msg)
    ends = np.asarray([(states[first], states[second]) for first, second in graph.edges()], dtype=float).reshape(-1, 2)
    return np.concatenate([ends[:, 0], ends[:, 1]]), np.concatenate([ends[:, 1], ends[:, 0]])


def _pairs(data: npt.ArrayLike | nx.Graph, attribute: str, *, diagonal: bool) -> tuple[Array, Array]:
    """Return the states of every agent and of each of its neighbours, on a grid or a network."""
    if isinstance(data, nx.Graph):
        return _graph_pairs(data, attribute)
    return _grid_pairs(np.asarray(data, dtype=float), diagonal=diagonal)


@analysis(DataKind.FIELD)
def like_neighbor_share(
    data: npt.ArrayLike | nx.Graph,
    *,
    attribute: str = 'state',
    empty: float | None = None,
    diagonal: bool = True,
) -> float:
    """Return the mean share of an agent's occupied neighbours that share its state, on a periodic grid or a network.

    Args:
        data:
            A grid of states, or a NetworkX graph whose nodes carry the states.

        attribute:
            The node attribute of the states of a graph.

        empty:
            The state of empty cells, left out as agents and as neighbours, or `None` when every cell is occupied.

        diagonal:
            Whether grid cells that share only a corner are neighbours, as in Schelling's Moore neighbourhood.

    Returns:
        share:
            The share, near the global share of a state for a random arrangement and near 1 when segregated.

    Raises:
        AnalysisError:
            A node of a graph has no state.

    Examples:
        >>> from skcomplex.core.analysis import like_neighbor_share
        >>> like_neighbor_share([[1, 1, 2, 2]] * 4, diagonal=False)
        0.75
    """
    own, other = _pairs(data, attribute, diagonal=diagonal)
    occupied = np.ones(own.size, dtype=bool) if empty is None else (own != empty) & (other != empty)
    agents = own[occupied]
    return float(np.mean(agents == other[occupied])) if agents.size else 0.0


@analysis(DataKind.FIELD)
def interface_density(data: npt.ArrayLike | nx.Graph, *, attribute: str = 'state') -> float:
    """Return the share of neighbouring pairs, on a periodic grid without corners or along edges, that differ.

    Args:
        data:
            A grid of states, or a NetworkX graph whose nodes carry the states.

        attribute:
            The node attribute of the states of a graph.

    Returns:
        density:
            0 for one uniform domain and 1/2 for random binary states.

    Raises:
        AnalysisError:
            A node of a graph has no state.

    Examples:
        >>> from skcomplex.core.analysis import interface_density
        >>> interface_density([[1, 1, -1, -1]] * 4)
        0.25
    """
    own, other = _pairs(data, attribute, diagonal=False)
    return float(np.mean(own != other))


@analysis(DataKind.AGENTS)
def dissimilarity_index(states: npt.ArrayLike, units: npt.ArrayLike) -> float:
    """Return Duncan's dissimilarity index of two groups across units, the share of one that would have to move.

    Args:
        states:
            The group of every agent, two distinct values.

        units:
            The unit of every agent, such as a district or a block of cells.

    Returns:
        index:
            `1/2 Σ_u |a_u / A - b_u / B|`, 0 for even mixing and 1 for complete separation.

    Raises:
        AnalysisError:
            The states do not hold exactly two groups.

    Examples:
        >>> from skcomplex.core.analysis import dissimilarity_index
        >>> dissimilarity_index([0, 0, 1, 1], [0, 0, 1, 1]), dissimilarity_index([0, 1, 0, 1], [0, 0, 1, 1])
        (1.0, 0.0)
    """
    groups = np.asarray(states)
    labels = np.unique(groups)
    if labels.size != 2:  # noqa: PLR2004
        error_msg = f'The dissimilarity index compares two groups, not {labels.size}.'
        raise AnalysisError(error_msg)
    _, places = np.unique(np.asarray(units), return_inverse=True)
    first = np.bincount(places[groups == labels[0]], minlength=places.max() + 1)
    second = np.bincount(places[groups == labels[1]], minlength=places.max() + 1)
    return float(0.5 * np.sum(np.abs(first / first.sum() - second / second.sum())))
