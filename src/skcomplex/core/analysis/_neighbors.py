"""Measure the neighbours of agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from ..engine.runtime import Trajectory
from ..enums import DataKind
from ..errors import AnalysisError
from ..types import Array, Indices
from ._analysis_tags import AnalysisTags
from ._base import SNAPSHOT_DIMENSIONS, _check_settings, _read_fields, _read_networks, _read_snapshots
from .types import DataSource

KINDS = ('distances', 'clark-evans', 'radial-distribution', 'radius', 'like-share')
SETTINGS = {
    'distances': {'group', 'box'},
    'clark-evans': {'group', 'box'},
    'radial-distribution': {'group', 'box', 'bins', 'r_max'},
    'radius': {'group', 'box', 'r_max'},
    'like-share': {'variables', 'attribute', 'empty', 'diagonal'},
}
TAGS = AnalysisTags(DataKind.AGENTS)
KIND_TAGS = {
    'like-share': AnalysisTags(DataKind.FIELD),
}


@dataclass(frozen=True)
class NeighborsResult:
    """A result of `neighbors`, the neighbours of agents, cells, or nodes.

    It holds the positions or the states it read and the statistic of the kind for every frame, such as a distance of
    1 between two points at 0.5 and 9.5 in a periodic box of side 10. It reports how close birds sit in a flock, how
    cells pack in a tissue, and how far households live from others like them.

    Read more in the user guide, under Analyse.

    Args:
        positions:
            The positions, frames by agents by dimensions, for the agent kinds.

        box:
            The sides of the periodic box, or `None` for open space, for the agent kinds.

        frames:
            The grids of states, or the graphs, for `'like-share'`.

        distances:
            The nearest-neighbour distances, frames by agents, for `'distances'`.

        ratio:
            One Clark-Evans ratio per frame, for `'clark-evans'`.

        radii:
            The bin centres of `g(r)`, for `'radial-distribution'`.

        distribution:
            `g(r)` at every bin centre, for `'radial-distribution'`.

        share:
            One like share per frame, for `'like-share'`.

        pairs:
            The pairs of agents within the radius, one array of rows `(i, j)` with `i < j` in increasing order per
            frame, for `'radius'`.

        degrees:
            The number of agents within the radius of every agent, frames by agents, for `'radius'`.

    See Also:
        neighbors: Returns this result.

    Examples:
        >>> from skcomplex.core.analysis import neighbors
        >>> neighbors([[1, 1, 2, 2]] * 4, kind='like-share', diagonal=False).share.tolist()
        [0.75]
    """

    positions: Array | None = None
    box: Array | None = None
    frames: list[Array] | list[nx.Graph] | None = None
    distances: Array | None = None
    ratio: Array | None = None
    radii: Array | None = None
    distribution: Array | None = None
    share: Array | None = None
    pairs: list[Indices] | None = None
    degrees: Indices | None = None


def neighbors(
    source: DataSource,
    *,
    kind: str = 'distances',
    group: str | None = None,
    variables: Sequence[str] | None = None,
    box: Sequence[float] | None = None,
    bins: int = 100,
    r_max: float | None = None,
    attribute: str = 'state',
    empty: float | None = None,
    diagonal: bool = True,
) -> NeighborsResult:
    """Return the neighbours of agents in space, of grid cells, or of network nodes.

    An array of positions holds one snapshot, one row per agent, or one snapshot per frame along its first axis, such
    as a distance of 1 between two points at 0.5 and 9.5 in a periodic box of side 10. A trajectory gives the positions
    of an agent group with the sides of its periodic space as the box, or its lattice frames for `'like-share'`. A grid
    array is one frame of states, and a graph carries the states on its nodes. It counts the neighbours of birds in a
    flock, of cells in a tissue, of particles in a liquid, and of households in a city.

    Read more in the user guide, under Analyse.

    Args:
        source:
            The trajectory, an array of positions or of states, or a graph.

        kind:
            What to return. `'distances'` gives the distance from every agent to its nearest neighbour, across periodic
            edges in a box. `'clark-evans'` gives the mean nearest neighbour distance over its value for uniform random
            points in a periodic box, 1 for random points, below 1 for clustered ones, and up to 2.15 for a triangular
            lattice in a plane. `'radial-distribution'` gives the radial distribution function `g(r)` in a periodic box,
            averaged over frames, 1 for an ideal gas. `'radius'` gives the pairs of agents within `r_max` of each
            other and the degree of every agent, across periodic edges in a box, which is the interaction graph of a
            flock at every frame. `'like-share'` gives the mean share of the occupied neighbours of
            an agent that share its state, on a periodic grid or along the edges of a network, near 1 when segregated.

        group:
            The path of the agent group of a trajectory, or `None` for the first one.

        variables:
            The lattice state of a trajectory whose frames `'like-share'` reads, or `None` for the first one.

        box:
            The sides of a periodic box, or `None` for the periodic space of a trajectory or for open space.

        bins:
            The number of distance bins of `g(r)`.

        r_max:
            The largest distance of `g(r)`, or `None` for half the smallest side, or the radius of `'radius'`, which
            needs one.

        attribute:
            The node attribute of the states of a graph.

        empty:
            The state of empty cells, left out as agents and as neighbours, or `None` when every cell is occupied.

        diagonal:
            Whether grid cells that share only a corner are neighbours, as in Schelling's Moore neighbourhood.

    Returns:
        result:
            The data read and the neighbour statistic of every frame.

    Raises:
        AnalysisError:
            The kind is unknown, a setting the kind does not read is changed, the number of bins is below 1, the agents
            have no positions, a kind that needs a periodic box has none, or `'radius'` has no positive `r_max`.

    See Also:
        clusters: Finds the clusters of a grid, a graph, or points.
        motifs: Counts the small patterns of links of a network.

    Examples:
        >>> from skcomplex.core.analysis import neighbors
        >>> neighbors([[0.5, 0.0], [9.5, 0.0]], box=[10.0, 10.0]).distances.tolist()
        [[1.0, 1.0]]
    """
    if kind not in KINDS:
        error_msg = f'neighbors has kind={kind!r}. Use one of {", ".join(KINDS)}.'
        raise AnalysisError(error_msg)
    _check_settings(
        neighbors,
        'kind',
        SETTINGS[kind],
        kind=kind,
        group=group,
        variables=variables,
        box=box,
        bins=bins,
        r_max=r_max,
        attribute=attribute,
        empty=empty,
        diagonal=diagonal,
    )
    if bins < 1:
        error_msg = f'neighbors has bins={bins!r}. Use 1 or more.'
        raise AnalysisError(error_msg)
    if kind == 'like-share':
        frames = _read_states(source, variables)
        shares = [_like_neighbor_share(frame, attribute=attribute, empty=empty, diagonal=diagonal) for frame in frames]
        return NeighborsResult(frames=frames, share=np.asarray(shares))
    positions, sides = _read_positions(source, group, box)
    if kind == 'distances':
        distances = np.asarray([_nearest_neighbor_distances(frame, sides) for frame in positions])
        return NeighborsResult(positions=positions, box=sides, distances=distances)
    if kind == 'radius':
        if r_max is None or r_max <= 0:
            error_msg = f'neighbors of kind {kind!r} has r_max={r_max!r}. Use a positive radius.'
            raise AnalysisError(error_msg)
        pairs = [_radius_pairs(frame, sides, r_max) for frame in positions]
        degrees = np.asarray([_degrees(found, positions.shape[1]) for found in pairs])
        return NeighborsResult(positions=positions, box=sides, pairs=pairs, degrees=degrees)
    if sides is None:
        error_msg = f'neighbors of kind {kind!r} needs a periodic box. Set `box` to its sides.'
        raise AnalysisError(error_msg)
    if kind == 'clark-evans':
        ratio = np.asarray([_clark_evans_ratio(frame, sides) for frame in positions])
        return NeighborsResult(positions=positions, box=sides, ratio=ratio)
    radii, distribution = _radial_distribution(positions, sides, bins, r_max)
    return NeighborsResult(positions=positions, box=sides, radii=radii, distribution=distribution)


def _read_states(source: DataSource, variables: Sequence[str] | None) -> list[Array] | list[nx.Graph]:
    """Return the grids of states of a trajectory or an array, or a graph."""
    if isinstance(source, nx.Graph):
        return _read_networks(source)
    name = variables[0] if variables else None
    frames = _read_fields(source, name)
    return list(frames) if isinstance(source, Trajectory) else [frames]


def _read_positions(source: DataSource, group: str | None, box: Sequence[float] | None) -> tuple[Array, Array | None]:
    """Return the positions and the box of the agents of a trajectory.

    An array of positions is also accepted.
    """
    sides = None if box is None else np.asarray(box, dtype=float)
    if isinstance(source, Trajectory):
        snapshots = _read_snapshots(source, group)
        if snapshots.positions is None:
            error_msg = f'The agents of {group or "the first group"} have no positions. Use agents in a space.'
            raise AnalysisError(error_msg)
        if sides is None and snapshots.periodic:
            sides = snapshots.box
        return snapshots.positions, sides
    array = np.asarray(source, dtype=float)
    return (array[None] if array.ndim == SNAPSHOT_DIMENSIONS else array), sides


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
    ends = np.asarray([(states[first], states[second]) for first, second in graph.edges()], dtype=float).reshape(
        -1,
        2,
    )
    return np.concatenate([ends[:, 0], ends[:, 1]]), np.concatenate([ends[:, 1], ends[:, 0]])


def _pairs(data: npt.ArrayLike | nx.Graph, attribute: str, *, diagonal: bool) -> tuple[Array, Array]:
    """Return the states of every agent and of each of its neighbours.

    The agents lie on a grid or a network.
    """
    if isinstance(data, nx.Graph):
        return _graph_pairs(data, attribute)
    return _grid_pairs(np.asarray(data, dtype=float), diagonal=diagonal)


def _tree(positions: npt.ArrayLike, box: npt.ArrayLike | None) -> tuple[Array, cKDTree]:
    """Return the positions and their tree, periodic in a box when one is given."""
    array = np.asarray(positions, dtype=float)
    array = array.reshape(array.shape[0], -1)
    if box is None:
        return array, cKDTree(array)
    sides = np.asarray(box, dtype=float)
    return np.mod(array, sides), cKDTree(np.mod(array, sides), boxsize=sides)


def _nearest_neighbor_distances(positions: npt.ArrayLike, box: npt.ArrayLike | None = None) -> Array:
    """Return the distance from every agent of one snapshot to its nearest neighbour.

    The distances go across periodic edges.
    """
    array, tree = _tree(positions, box)
    return tree.query(array, k=2)[0][:, 1]


def _radius_pairs(positions: npt.ArrayLike, box: npt.ArrayLike | None, r_max: float) -> Indices:
    """Return the pairs of agents of one snapshot within a radius of each other.

    The distances go across periodic edges, and the pairs come in increasing order.
    """
    _, tree = _tree(positions, box)
    pairs = np.asarray(tree.query_pairs(r_max, output_type='ndarray'), dtype=np.intp).reshape(-1, 2)
    return np.unique(pairs, axis=0).reshape(-1, 2)


def _degrees(pairs: Indices, size: int) -> Indices:
    """Return the number of pairs every agent belongs to."""
    return np.bincount(pairs.ravel(), minlength=size)


def _clark_evans_ratio(positions: npt.ArrayLike, box: npt.ArrayLike) -> float:
    """Return the mean nearest-neighbour distance of a snapshot over its random value.

    The random value is that of uniform random points.
    """
    sides = np.asarray(box, dtype=float)
    distances = _nearest_neighbor_distances(positions, sides)
    dimension = sides.size
    density = distances.size / np.prod(sides)
    ball = math.pi ** (dimension / 2) / math.gamma(dimension / 2 + 1)
    expected = math.gamma(1 / dimension + 1) / (density * ball) ** (1 / dimension)
    return float(distances.mean() / expected)


def _radial_distribution(
    snapshots: Array,
    box: npt.ArrayLike,
    bins: int,
    r_max: float | None,
) -> tuple[Array, Array]:
    """Return the bin centres and `g(r)` of frames of positions in a periodic box.

    The values are averaged over frames.
    """
    sides = np.asarray(box, dtype=float)
    reach = float(sides.min() / 2) if r_max is None else r_max
    edges = np.linspace(0, reach, bins + 1)
    counts = np.zeros(bins)
    for snapshot in snapshots:
        _, tree = _tree(snapshot, sides)
        pairs = tree.sparse_distance_matrix(tree, reach, output_type='ndarray')
        distances = pairs['v'][pairs['i'] != pairs['j']]
        counts += np.histogram(distances, bins=edges)[0]
    dimension = sides.size
    ball = math.pi ** (dimension / 2) / math.gamma(dimension / 2 + 1)
    shells = ball * (edges[1:] ** dimension - edges[:-1] ** dimension)
    size = snapshots.shape[1]
    ideal = snapshots.shape[0] * size * (size - 1) / np.prod(sides) * shells
    return (edges[1:] + edges[:-1]) / 2, counts / ideal


def _like_neighbor_share(
    data: npt.ArrayLike | nx.Graph,
    *,
    attribute: str = 'state',
    empty: float | None = None,
    diagonal: bool = True,
) -> float:
    """Return the mean share of an agent's occupied neighbours that share its state.

    The agents lie on a grid or a network.
    """
    own, other = _pairs(data, attribute, diagonal=diagonal)
    occupied = np.ones(own.size, dtype=bool) if empty is None else (own != empty) & (other != empty)
    agents = own[occupied]
    return float(np.mean(agents == other[occupied])) if agents.size else 0.0
