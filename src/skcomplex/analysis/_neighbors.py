"""Measure the neighbours of agents: nearest-neighbour distances, the Clark-Evans ratio, `g(r)`, and like shares."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from typing import Self

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from ..core.errors import AnalysisError
from ..engine.runtime import Trajectory
from ._base import BaseAnalysis, Source
from ._kinds import DataKind

type Array = npt.NDArray[np.float64]
KINDS = ('distances', 'clark-evans', 'radial-distribution', 'like-share')
SETTINGS = {
    'distances': {'group', 'box'},
    'clark-evans': {'group', 'box'},
    'radial-distribution': {'group', 'box', 'bins', 'r_max'},
    'like-share': {'variables', 'attribute', 'empty', 'diagonal'},
}
"""Settings every kind reads, besides `kind`."""


class NeighborAnalysis(BaseAnalysis):
    """Neighbours of agents in space, or of cells of a grid or nodes of a network, of one kind.

    The kinds read positions, or states on grids and networks:

    - `'distances'`: the distance from every agent to its nearest neighbour, across periodic edges in a box.
    - `'clark-evans'`: the mean nearest-neighbour distance over its value for uniform random points in a periodic box,
      1 for random points, below 1 for clustered ones, and up to 2.15 for a triangular lattice in a plane.
    - `'radial-distribution'`: the radial distribution function `g(r)` in a periodic box, averaged over frames, 1 for
      an ideal gas.
    - `'like-share'`: the mean share of an agent's occupied neighbours that share its state, on a periodic grid or
      along the edges of a network, near 1 when segregated.

    An array of positions holds one snapshot, one row per agent, or one snapshot per frame along its first axis. A
    trajectory gives the positions of an agent group with the sides of its periodic space as the box, or its lattice
    frames for `'like-share'`. A grid array is one frame of states, and a graph carries the states on its nodes.

    Args:
        kind:
            The neighbour statistic to compute, one of the kinds above.

        group:
            The path of the agent group of a trajectory, or `None` for the first one.

        variables:
            The lattice state of a trajectory whose frames `'like-share'` reads, or `None` for the first one.

        box:
            The sides of a periodic box, or `None` for the periodic space of a trajectory or for open space.

        bins:
            The number of distance bins of `g(r)`.

        r_max:
            The largest distance of `g(r)`, or `None` for half the smallest side.

        attribute:
            The node attribute of the states of a graph.

        empty:
            The state of empty cells, left out as agents and as neighbours, or `None` when every cell is occupied.

        diagonal:
            Whether grid cells that share only a corner are neighbours, as in Schelling's Moore neighbourhood.

    Attributes:
        positions_:
            The positions, frames by agents by dimensions, for the agent kinds.

        box_:
            The sides of the periodic box, or `None` for open space.

        frames_:
            The grids of states, or the graphs, of `'like-share'`.

        distances_:
            The nearest-neighbour distances, frames by agents, of `'distances'`.

        ratio_:
            One Clark-Evans ratio per frame.

        radii_:
            The bin centres of `g(r)`.

        distribution_:
            `g(r)` at every bin centre.

        share_:
            One like share per frame.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import NeighborAnalysis
        >>> positions = [[0.5, 0.0], [9.5, 0.0]]
        >>> NeighborAnalysis(box=[10.0, 10.0]).check().prepare(positions).run().distances_.tolist()
        [[1.0, 1.0]]
        >>> points = np.random.default_rng(0).random((5000, 2))
        >>> NeighborAnalysis('clark-evans', box=[1.0, 1.0]).check().prepare(points).run().ratio_.round(1).tolist()
        [1.0]
        >>> NeighborAnalysis('like-share', diagonal=False).check().prepare([[1, 1, 2, 2]] * 4).run().share_.tolist()
        [0.75]
    """

    def __init__(
        self,
        kind: str = 'distances',
        group: str | None = None,
        *,
        variables: Sequence[str] | None = None,
        box: Sequence[float] | None = None,
        bins: int = 100,
        r_max: float | None = None,
        attribute: str = 'state',
        empty: float | None = None,
        diagonal: bool = True,
    ) -> None:
        self.kind = kind
        self.group = group
        self.variables = variables
        self.box = box
        self.bins = bins
        self.r_max = r_max
        self.attribute = attribute
        self.empty = empty
        self.diagonal = diagonal

    def _check(self) -> Self:
        """Validate the kind, the number of bins, and that the kind reads every changed setting."""
        if self.kind not in KINDS:
            error_msg = f'{type(self).__name__} has kind={self.kind!r}. Use one of {", ".join(KINDS)}.'
            raise AnalysisError(error_msg)
        self._check_settings('kind', SETTINGS[self.kind])
        if self.bins < 1:
            error_msg = f'{type(self).__name__} has bins={self.bins!r}. Use 1 or more.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the field kind for like shares, and the agents kind otherwise."""
        return DataKind.FIELD if self.kind == 'like-share' else DataKind.AGENTS

    def _prepare_states(self, source: Source) -> Self:
        """Read the grids of states of a trajectory or an array, or a graph."""
        if isinstance(source, nx.Graph):
            self.frames_ = self._read_networks(source)
            return self
        name = self.variables[0] if self.variables else None
        frames = self._read_fields(source, name)
        self.frames_ = list(frames) if isinstance(source, Trajectory) else [frames]
        return self

    def _prepare(self, source: Source) -> Self:
        """Read the positions and the box of the agents, or the states of a grid or a network."""
        if self.kind == 'like-share':
            return self._prepare_states(source)
        box = None if self.box is None else np.asarray(self.box, dtype=float)
        if isinstance(source, Trajectory):
            snapshots = self._read_snapshots(source, self.group)
            if snapshots.positions is None:
                error_msg = f'The agents of {self.group or "the first group"} have no positions. Use agents in a space.'
                raise AnalysisError(error_msg)
            positions = snapshots.positions
            if box is None and snapshots.periodic:
                box = snapshots.box
        else:
            array = np.asarray(source, dtype=float)
            positions = array[None] if array.ndim == 2 else array  # noqa: PLR2004
        self.positions_ = positions
        self.box_ = box
        return self

    def _run(self, inputs: object = None) -> Self:
        """Compute the neighbour statistic of every frame."""
        if self.kind == 'like-share':
            shares = [
                self._like_neighbor_share(frame, attribute=self.attribute, empty=self.empty, diagonal=self.diagonal)
                for frame in self.frames_
            ]
            self.share_ = np.asarray(shares)
            return self
        if self.kind == 'distances':
            self.distances_ = np.asarray(
                [self._nearest_neighbor_distances(frame, self.box_) for frame in self.positions_],
            )
            return self
        if self.box_ is None:
            error_msg = f'{type(self).__name__} of kind {self.kind!r} needs a periodic box. Set `box` to its sides.'
            raise AnalysisError(error_msg)
        if self.kind == 'clark-evans':
            self.ratio_ = np.asarray([self._clark_evans_ratio(frame, self.box_) for frame in self.positions_])
            return self
        self.radii_, self.distribution_ = self._radial_distribution(self.positions_, self.box_, self.bins, self.r_max)
        return self

    @staticmethod
    def _shifts(ndim: int, *, diagonal: bool) -> list[tuple[int, ...]]:
        """Return the offsets of the neighbours of a cell, with or without corners."""
        offsets = np.stack(np.meshgrid(*([[-1, 0, 1]] * ndim), indexing='ij'), axis=-1).reshape(-1, ndim)
        steps = np.abs(offsets).sum(axis=1)
        keep = (steps > 0) if diagonal else (steps == 1)
        return [tuple(int(item) for item in offset) for offset in offsets[keep]]

    @staticmethod
    def _grid_pairs(grid: npt.NDArray[np.float64], *, diagonal: bool) -> tuple[Array, Array]:
        """Return the states of every cell paired with every neighbour on a periodic grid."""
        pairs = [
            (grid, np.roll(grid, shift, axis=tuple(range(grid.ndim))))
            for shift in NeighborAnalysis._shifts(grid.ndim, diagonal=diagonal)
        ]
        return np.concatenate([first.ravel() for first, _ in pairs]), np.concatenate(
            [second.ravel() for _, second in pairs],
        )

    @staticmethod
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

    @staticmethod
    def _pairs(data: npt.ArrayLike | nx.Graph, attribute: str, *, diagonal: bool) -> tuple[Array, Array]:
        """Return the states of every agent and of each of its neighbours, on a grid or a network."""
        if isinstance(data, nx.Graph):
            return NeighborAnalysis._graph_pairs(data, attribute)
        return NeighborAnalysis._grid_pairs(np.asarray(data, dtype=float), diagonal=diagonal)

    @staticmethod
    def _tree(positions: npt.ArrayLike, box: npt.ArrayLike | None) -> tuple[Array, cKDTree]:
        """Return the positions and their tree, periodic in a box when one is given."""
        array = np.asarray(positions, dtype=float)
        array = array.reshape(array.shape[0], -1)
        if box is None:
            return array, cKDTree(array)
        sides = np.asarray(box, dtype=float)
        return np.mod(array, sides), cKDTree(np.mod(array, sides), boxsize=sides)

    @staticmethod
    def _nearest_neighbor_distances(positions: npt.ArrayLike, box: npt.ArrayLike | None = None) -> Array:
        """Return the distance from every agent of one snapshot to its nearest neighbour, across periodic edges."""
        array, tree = NeighborAnalysis._tree(positions, box)
        return tree.query(array, k=2)[0][:, 1]

    @staticmethod
    def _clark_evans_ratio(positions: npt.ArrayLike, box: npt.ArrayLike) -> float:
        """Return the mean nearest-neighbour distance of one snapshot over its value for uniform random points."""
        sides = np.asarray(box, dtype=float)
        distances = NeighborAnalysis._nearest_neighbor_distances(positions, sides)
        dimension = sides.size
        density = distances.size / np.prod(sides)
        ball = math.pi ** (dimension / 2) / math.gamma(dimension / 2 + 1)
        expected = math.gamma(1 / dimension + 1) / (density * ball) ** (1 / dimension)
        return float(distances.mean() / expected)

    @staticmethod
    def _radial_distribution(
        snapshots: Array,
        box: npt.ArrayLike,
        bins: int,
        r_max: float | None,
    ) -> tuple[Array, Array]:
        """Return the bin centres and `g(r)` of frames of positions in a periodic box, averaged over frames."""
        sides = np.asarray(box, dtype=float)
        reach = float(sides.min() / 2) if r_max is None else r_max
        edges = np.linspace(0, reach, bins + 1)
        counts = np.zeros(bins)
        for snapshot in snapshots:
            _, tree = NeighborAnalysis._tree(snapshot, sides)
            pairs = tree.sparse_distance_matrix(tree, reach, output_type='ndarray')
            distances = pairs['v'][pairs['i'] != pairs['j']]
            counts += np.histogram(distances, bins=edges)[0]
        dimension = sides.size
        ball = math.pi ** (dimension / 2) / math.gamma(dimension / 2 + 1)
        shells = ball * (edges[1:] ** dimension - edges[:-1] ** dimension)
        size = snapshots.shape[1]
        ideal = snapshots.shape[0] * size * (size - 1) / np.prod(sides) * shells
        return (edges[1:] + edges[:-1]) / 2, counts / ideal

    @staticmethod
    def _like_neighbor_share(
        data: npt.ArrayLike | nx.Graph,
        *,
        attribute: str = 'state',
        empty: float | None = None,
        diagonal: bool = True,
    ) -> float:
        """Return the mean share of an agent's occupied neighbours that share its state, on a grid or a network."""
        own, other = NeighborAnalysis._pairs(data, attribute, diagonal=diagonal)
        occupied = np.ones(own.size, dtype=bool) if empty is None else (own != empty) & (other != empty)
        agents = own[occupied]
        return float(np.mean(agents == other[occupied])) if agents.size else 0.0
