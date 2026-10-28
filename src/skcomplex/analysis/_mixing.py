"""Measure how categorical states mix or cluster on grids and networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
from collections.abc import Sequence
from typing import Self

import networkx as nx
import numpy as np
import numpy.typing as npt

from ..core.errors import AnalysisError
from ..engine.runtime import Trajectory
from ._base import BaseAnalysis, Source
from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]
KINDS = ('like-neighbors', 'dissimilarity')
SETTINGS = {
    'like-neighbors': {'attribute', 'empty', 'diagonal'},
    'dissimilarity': set(),
}
"""Settings every kind reads, besides `kind` and `variables`."""


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
        >>> from skcomplex.analysis import interface_density
        >>> interface_density([[1, 1, -1, -1]] * 4)
        0.25
    """
    own, other = _pairs(data, attribute, diagonal=False)
    return float(np.mean(own != other))


class MixingAnalysis(BaseAnalysis):
    """Mixing of categorical states of agents on a grid, a network, or across units, of one kind.

    - `'like-neighbors'`: the mean share of an agent's occupied neighbours that share its state, on a periodic grid or
      along the edges of a network, near the global share of a state for a random arrangement and near 1 when
      segregated.
    - `'dissimilarity'`: Duncan's dissimilarity index of two groups across units, `1/2 Σ_u |a_u / A - b_u / B|`, the
      share of one group that would have to move, 0 for even mixing and 1 for complete separation. It reads one row
      per agent with its group and its unit, such as a district or a block of cells.

    Args:
        kind:
            The measure, one of the kinds above.

        variables:
            The lattice state of a trajectory whose last frame `'like-neighbors'` reads, or `None` for the first one.

        attribute:
            The node attribute of the states of a graph.

        empty:
            The state of empty cells, left out as agents and as neighbours, or `None` when every cell is occupied.

        diagonal:
            Whether grid cells that share only a corner are neighbours, as in Schelling's Moore neighbourhood.

    Attributes:
        values_:
            The grid of states, the graph whose nodes carry the states, or the group and the unit of every agent.

        like_neighbor_share_:
            The share of like neighbours, of `'like-neighbors'`.

        dissimilarity_:
            The dissimilarity index, of `'dissimilarity'`.

    Examples:
        >>> from skcomplex.analysis import MixingAnalysis
        >>> mixing = MixingAnalysis('like-neighbors', diagonal=False)
        >>> mixing.check().prepare([[1, 1, 2, 2]] * 4).run().like_neighbor_share_
        0.75
        >>> MixingAnalysis('dissimilarity').check().prepare([[0, 0], [0, 0], [1, 1], [1, 1]]).run().dissimilarity_
        1.0
    """

    def __init__(
        self,
        kind: str = 'like-neighbors',
        variables: Sequence[str] | None = None,
        *,
        attribute: str = 'state',
        empty: float | None = None,
        diagonal: bool = True,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.attribute = attribute
        self.empty = empty
        self.diagonal = diagonal

    def _check(self) -> Self:
        """Validate the kind and that the kind reads every changed setting."""
        if self.kind not in KINDS:
            error_msg = f'{type(self).__name__} has kind={self.kind!r}. Use one of {", ".join(KINDS)}.'
            raise AnalysisError(error_msg)
        defaults = {
            name: parameter.default for name, parameter in inspect.signature(type(self).__init__).parameters.items()
        }
        read = SETTINGS[self.kind] | {'kind', 'variables'}
        ignored = [
            name
            for name, value in self.get_params().items()
            if name not in read and value is not defaults[name] and (defaults[name] is None or value != defaults[name])
        ]
        if ignored:
            error_msg = (
                f'{type(self).__name__} of kind {self.kind!r} does not read {", ".join(ignored)}. It reads '
                f'{", ".join(sorted(SETTINGS[self.kind]))}.'
            )
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the field kind for neighbours, and the agents kind for units."""
        return DataKind.FIELD if self.kind == 'like-neighbors' else DataKind.AGENTS

    def _prepare(self, source: Source) -> Self:
        """Read a graph, the last frame of a field, or the groups and units of the agents."""
        if isinstance(source, nx.Graph):
            self.values_ = source
            return self
        if self.kind == 'like-neighbors':
            name = self.variables[0] if self.variables else None
            frames = self._read_fields(source, name)
            self.values_ = frames[-1] if isinstance(source, Trajectory) else frames
            return self
        self.values_ = np.asarray(source)
        if self.values_.ndim != 2 or self.values_.shape[1] != 2:  # noqa: PLR2004
            error_msg = (
                f'{type(self).__name__} of kind dissimilarity reads one row per agent with its group and its unit, '
                f'not an array of shape {self.values_.shape}.'
            )
            raise AnalysisError(error_msg)
        return self

    def _run_dissimilarity(self) -> None:
        """Compute Duncan's dissimilarity index of the two groups."""
        groups = self.values_[:, 0]
        labels = np.unique(groups)
        if labels.size != 2:  # noqa: PLR2004
            error_msg = f'The dissimilarity index compares two groups, not {labels.size}. Give agents of two groups.'
            raise AnalysisError(error_msg)
        _, places = np.unique(self.values_[:, 1], return_inverse=True)
        first = np.bincount(places[groups == labels[0]], minlength=places.max() + 1)
        second = np.bincount(places[groups == labels[1]], minlength=places.max() + 1)
        self.dissimilarity_ = float(0.5 * np.sum(np.abs(first / first.sum() - second / second.sum())))

    def _run(self) -> Self:
        """Compute the measure of the kind."""
        if self.kind == 'dissimilarity':
            self._run_dissimilarity()
            return self
        own, other = _pairs(self.values_, self.attribute, diagonal=self.diagonal)
        occupied = np.ones(own.size, dtype=bool) if self.empty is None else (own != self.empty) & (other != self.empty)
        agents = own[occupied]
        self.like_neighbor_share_ = float(np.mean(agents == other[occupied])) if agents.size else 0.0
        return self
