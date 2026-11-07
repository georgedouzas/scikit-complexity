"""Quantify recurrences of embedded series: recurrence matrices, line structures, and their statistics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import sparse, stats
from scipy.spatial import cKDTree

from ..errors import AnalysisError
from ..types import Array
from ._base import _read_series
from .types import Source

PAIR_SAMPLE = 200_000


@dataclass(frozen=True)
class RecurrenceResult:
    """Recurrence matrix of the points of a source and its quantification.

    Args:
        values:
            The points, one per row.

        labels:
            The label of every coordinate.

        radius:
            The recurrence radius.

        matrix:
            The symmetric sparse boolean recurrence matrix, with a true diagonal.

        rate:
            The share of recurrent pairs off the Theiler window.

        determinism:
            The share of recurrent points on diagonal lines of at least the minimum length.

        laminarity:
            The share of recurrent points on vertical lines of at least the minimum length.

        mean_diagonal:
            The mean length of those diagonal lines.

        longest_diagonal:
            The length of the longest diagonal line.

        trapping_time:
            The mean length of those vertical lines.

        entropy:
            The Shannon entropy in nats of the distribution of those diagonal line lengths.
    """

    values: Array
    labels: list[str]
    radius: float
    matrix: sparse.csr_array
    rate: float
    determinism: float
    laminarity: float
    mean_diagonal: float
    longest_diagonal: int
    trapping_time: float
    entropy: float


def recurrence(
    source: Source,
    *,
    variables: Sequence[str] | None = None,
    radius: float | None = None,
    rate: float = 0.05,
    min_line: int = 2,
    theiler: int = 1,
    random_state: int | None = 0,
) -> RecurrenceResult:
    """Return the recurrence matrix of the points of a source and its quantification.

    Every row of the series of a source is a point, and every series a coordinate, so an embedded series or the
    states of a trajectory give a point set. Two points recur when they lie within the radius of each other.

    Args:
        source:
            The trajectory, or an array with one row per point and one column per coordinate.

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

    Returns:
        result:
            The points, the recurrence matrix, and its quantification.

    Raises:
        AnalysisError:
            A setting is out of range, or no pair of points recurs outside the Theiler window.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import embedding, recurrence
        >>> sine = embedding(np.sin(np.arange(2000) * 0.1), delay=15, dimension=2)
        >>> bool(recurrence(sine.points).determinism > 0.95)
        True
        >>> recurrence([0.0, 0.1, 5.0, 5.1], radius=0.5).matrix.toarray().astype(int).tolist()
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]
    """
    if (radius is not None and radius <= 0) or not 0 < rate < 1:
        error_msg = (
            f'recurrence has radius={radius!r} and rate={rate!r}. Use a positive radius or None, and a rate between 0 '
            'and 1.'
        )
        raise AnalysisError(error_msg)
    if min_line < 1 or theiler < 0:
        error_msg = (
            f'recurrence has min_line={min_line!r} and theiler={theiler!r}. Use a shortest line of 1 or more and a '
            'Theiler window of 0 or more.'
        )
        raise AnalysisError(error_msg)
    series = _read_series(source, variables)
    array = series.values
    chosen = _radius_for_rate(array, rate, random_state) if radius is None else radius
    recurrences = _recurrence_matrix(array, chosen)
    matrix = recurrences.tocoo()
    rows, columns = matrix.row.astype(np.int64), matrix.col.astype(np.int64)
    outside = np.abs(rows - columns) >= theiler
    rows, columns = rows[outside], columns[outside]
    if not rows.size:
        error_msg = (
            f'No pair of points recurs within radius {chosen} outside the Theiler window {theiler}. Use a larger '
            'radius or rate.'
        )
        raise AnalysisError(error_msg)
    diagonals = _run_lengths(columns - rows, rows)
    verticals = _run_lengths(columns, rows)
    long_diagonals = diagonals[diagonals >= min_line]
    long_verticals = verticals[verticals >= min_line]
    _, counts = np.unique(long_diagonals, return_counts=True)
    size = array.shape[0]
    allowed = size * size - size - 2 * sum(size - offset for offset in range(1, theiler))
    return RecurrenceResult(
        values=array,
        labels=series.labels,
        radius=chosen,
        matrix=recurrences,
        rate=float(rows.size / max(allowed, 1)),
        determinism=_line_share(diagonals, min_line, rows.size),
        laminarity=_line_share(verticals, min_line, rows.size),
        mean_diagonal=float(long_diagonals.mean()) if long_diagonals.size else 0.0,
        longest_diagonal=int(diagonals.max()),
        trapping_time=float(long_verticals.mean()) if long_verticals.size else 0.0,
        entropy=float(stats.entropy(counts)) if counts.size else 0.0,
    )


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


def _recurrence_matrix(array: Array, radius: float) -> sparse.csr_array:
    """Return the symmetric boolean recurrence matrix of points, true where two lie within a radius."""
    tree = cKDTree(array)
    pairs = tree.query_pairs(radius, output_type='ndarray')
    size = array.shape[0]
    rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(size)])
    columns = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(size)])
    return sparse.csr_array((np.ones(rows.size, dtype=bool), (rows, columns)), shape=(size, size))


def _run_lengths(keys: npt.NDArray[np.int64], positions: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Return the lengths of runs of consecutive positions that share a key."""
    if not keys.size:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((positions, keys))
    keys, positions = keys[order], positions[order]
    breaks = np.flatnonzero((np.diff(keys) != 0) | (np.diff(positions) != 1)) + 1
    edges = np.concatenate([[0], breaks, [keys.size]])
    return np.diff(edges)


def _line_share(lengths: npt.NDArray[np.int64], minimum: int, total: int) -> float:
    """Return the share of recurrent points on lines at least `minimum` long."""
    return float(lengths[lengths >= minimum].sum() / total) if total else 0.0
