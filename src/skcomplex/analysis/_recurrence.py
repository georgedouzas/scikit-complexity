"""Quantify recurrences of embedded series: recurrence matrices, line structures, and their statistics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import sparse, stats
from scipy.spatial import cKDTree

from ..errors import AnalysisError
from ..types import Array, DataKind
from ._base import BaseAnalysis
from .types import Source

PAIR_SAMPLE = 200_000


class RecurrenceAnalysis(BaseAnalysis):
    """Recurrence quantification of the points of a source: rate, determinism, laminarity, lines, and entropy.

    Every row of the series of a source is a point, and every series a coordinate, so an embedded series or the
    states of a trajectory give a point set. Two points recur when they lie within the radius of each other.

    Args:
        variables:
            The variables of a trajectory to read as coordinates, or `None` for every scalar series.

        radius:
            The recurrence radius, or `None` to choose the radius that gives the recurrence rate.

        rate:
            The share of recurrent pairs when the radius is not given.

        min_line:
            The shortest diagonal or vertical line counted.

        theiler:
            The number of diagonals about the main one left out, 1 to leave out the main diagonal alone.

        random_state:
            The seed of the sample of pairs that sets the radius of large sets.

    Attributes:
        values_:
            The points, one per row.

        labels_:
            The label of every coordinate.

        radius_:
            The recurrence radius.

        matrix_:
            The symmetric sparse boolean recurrence matrix, with a true diagonal.

        rate_:
            The share of recurrent pairs off the Theiler window.

        determinism_:
            The share of recurrent points on diagonal lines of at least the minimum length.

        laminarity_:
            The share of recurrent points on vertical lines of at least the minimum length.

        mean_diagonal_:
            The mean length of those diagonal lines.

        longest_diagonal_:
            The length of the longest diagonal line.

        trapping_time_:
            The mean length of those vertical lines.

        entropy_:
            The Shannon entropy in nats of the distribution of those diagonal line lengths.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import EmbeddingAnalysis, RecurrenceAnalysis
        >>> sine = EmbeddingAnalysis(delay=15, dimension=2).check().prepare(np.sin(np.arange(2000) * 0.1)).run()
        >>> bool(RecurrenceAnalysis().check().prepare(sine.points_).run().determinism_ > 0.95)
        True
        >>> recurrence = RecurrenceAnalysis(radius=0.5).check().prepare([0.0, 0.1, 5.0, 5.1]).run()
        >>> recurrence.matrix_.toarray().astype(int).tolist()
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]
    """

    def __init__(
        self,
        variables: Sequence[str] | None = None,
        *,
        radius: float | None = None,
        rate: float = 0.05,
        min_line: int = 2,
        theiler: int = 1,
        random_state: int | None = 0,
    ) -> None:
        self.variables = variables
        self.radius = radius
        self.rate = rate
        self.min_line = min_line
        self.theiler = theiler
        self.random_state = random_state

    def _check(self) -> None:
        """Validate the radius, the rate, the shortest line, and the Theiler window."""
        if (self.radius is not None and self.radius <= 0) or not 0 < self.rate < 1:
            error_msg = (
                f'{type(self).__name__} has radius={self.radius!r} and rate={self.rate!r}. Use a positive radius or '
                'None, and a rate between 0 and 1.'
            )
            raise AnalysisError(error_msg)
        if self.min_line < 1 or self.theiler < 0:
            error_msg = (
                f'{type(self).__name__} has min_line={self.min_line!r} and theiler={self.theiler!r}. Use a shortest '
                'line of 1 or more and a Theiler window of 0 or more.'
            )
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the series kind, whose rows are the points."""
        return DataKind.SERIES

    def _prepare(self, source: Source) -> None:
        """Read the series as the coordinates of the points."""
        series = self._read_series(source, self.variables)
        self.values_ = series.values
        self.labels_ = series.labels

    def _run(self, inputs: object = None) -> None:
        """Compute the recurrence matrix and its quantification."""
        array = self.values_
        self.radius_ = (
            self._radius_for_rate(array, self.rate, self.random_state) if self.radius is None else self.radius
        )
        self.matrix_ = self._recurrence_matrix(array, self.radius_)
        matrix = self.matrix_.tocoo()
        rows, columns = matrix.row.astype(np.int64), matrix.col.astype(np.int64)
        outside = np.abs(rows - columns) >= self.theiler
        rows, columns = rows[outside], columns[outside]
        if not rows.size:
            error_msg = (
                f'No pair of points recurs within radius {self.radius_} outside the Theiler window {self.theiler}. '
                'Use a larger radius or rate.'
            )
            raise AnalysisError(error_msg)
        diagonals = self._run_lengths(columns - rows, rows)
        verticals = self._run_lengths(columns, rows)
        long_diagonals = diagonals[diagonals >= self.min_line]
        long_verticals = verticals[verticals >= self.min_line]
        _, counts = np.unique(long_diagonals, return_counts=True)
        size = array.shape[0]
        allowed = size * size - size - 2 * sum(size - offset for offset in range(1, self.theiler))
        self.rate_ = float(rows.size / max(allowed, 1))
        self.determinism_ = self._line_share(diagonals, self.min_line, rows.size)
        self.laminarity_ = self._line_share(verticals, self.min_line, rows.size)
        self.mean_diagonal_ = float(long_diagonals.mean()) if long_diagonals.size else 0.0
        self.longest_diagonal_ = int(diagonals.max())
        self.trapping_time_ = float(long_verticals.mean()) if long_verticals.size else 0.0
        self.entropy_ = float(stats.entropy(counts)) if counts.size else 0.0

    @staticmethod
    def _radius_for_rate(array: Array, rate: float, random_state: int | None) -> float:
        """Return the radius at which the share of recurrent pairs is the given rate."""
        size = array.shape[0]
        if size * (size - 1) // 2 <= PAIR_SAMPLE:
            first, second = np.triu_indices(size, k=1)
        else:
            rng = np.random.default_rng(random_state)
            first, second = rng.integers(0, size, (2, PAIR_SAMPLE))
            keep = first != second
            first, second = first[keep], second[keep]
        return float(np.quantile(np.linalg.norm(array[first] - array[second], axis=1), rate))

    @staticmethod
    def _recurrence_matrix(array: Array, radius: float) -> sparse.csr_array:
        """Return the symmetric boolean recurrence matrix of points, true where two lie within a radius."""
        tree = cKDTree(array)
        pairs = tree.query_pairs(radius, output_type='ndarray')
        size = array.shape[0]
        rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(size)])
        columns = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(size)])
        return sparse.csr_array((np.ones(rows.size, dtype=bool), (rows, columns)), shape=(size, size))

    @staticmethod
    def _run_lengths(keys: npt.NDArray[np.int64], positions: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """Return the lengths of runs of consecutive positions that share a key."""
        if not keys.size:
            return np.zeros(0, dtype=np.int64)
        order = np.lexsort((positions, keys))
        keys, positions = keys[order], positions[order]
        breaks = np.flatnonzero((np.diff(keys) != 0) | (np.diff(positions) != 1)) + 1
        edges = np.concatenate([[0], breaks, [keys.size]])
        return np.diff(edges)

    @staticmethod
    def _line_share(lengths: npt.NDArray[np.int64], minimum: int, total: int) -> float:
        """Return the share of recurrent points on lines at least `minimum` long."""
        return float(lengths[lengths >= minimum].sum() / total) if total else 0.0
