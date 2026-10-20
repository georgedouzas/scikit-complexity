"""Quantify recurrences of embedded series: recurrence matrices, line structures, and their statistics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.spatial import cKDTree

from ..base import AnalysisError
from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]
PAIR_SAMPLE = 200_000


def _points(points: npt.ArrayLike) -> Array:
    """Return points as one row per point."""
    array = np.asarray(points, dtype=float)
    return array.reshape(array.shape[0], -1)


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


@analysis(DataKind.SERIES)
def recurrence_matrix(
    points: npt.ArrayLike,
    radius: float | None = None,
    *,
    rate: float = 0.05,
    random_state: int | None = 0,
) -> sparse.csr_array:
    """Return the sparse recurrence matrix of points, true where two points lie within a radius of each other.

    Args:
        points:
            One row per point of an embedded series, or one value per point.

        radius:
            The recurrence radius, or `None` to choose the radius that gives the recurrence rate.

        rate:
            The share of recurrent pairs when the radius is not given.

        random_state:
            The seed of the sample of pairs that sets the radius of large sets.

    Returns:
        matrix:
            The symmetric boolean recurrence matrix, with a true diagonal.

    Examples:
        >>> from skcomplex.core.analysis import recurrence_matrix
        >>> recurrence_matrix([0.0, 0.1, 5.0], 0.5).toarray().astype(int).tolist()
        [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
    """
    array = _points(points)
    chosen = _radius_for_rate(array, rate, random_state) if radius is None else radius
    tree = cKDTree(array)
    pairs = tree.query_pairs(chosen, output_type='ndarray')
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


@dataclass(frozen=True)
class Recurrence:
    """Recurrence quantification of a point set.

    Args:
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

        radius:
            The recurrence radius.
    """

    rate: float
    determinism: float
    laminarity: float
    mean_diagonal: float
    longest_diagonal: int
    trapping_time: float
    entropy: float
    radius: float


@analysis(DataKind.SERIES)
def recurrence_quantification(
    points: npt.ArrayLike,
    radius: float | None = None,
    *,
    rate: float = 0.05,
    min_line: int = 2,
    theiler: int = 1,
    random_state: int | None = 0,
) -> Recurrence:
    """Return the recurrence quantification analysis of points: rate, determinism, laminarity, lines, and entropy.

    Args:
        points:
            One row per point of an embedded series, or one value per point.

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
        recurrence:
            The recurrence measures.

    Raises:
        AnalysisError:
            No pair of points recurs outside the Theiler window.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import delay_embedding, recurrence_quantification
        >>> sine = delay_embedding(np.sin(np.arange(2000) * 0.1), 2, 15)
        >>> bool(recurrence_quantification(sine).determinism > 0.95)
        True
    """
    array = _points(points)
    chosen = _radius_for_rate(array, rate, random_state) if radius is None else radius
    matrix = recurrence_matrix(array, chosen).tocoo()
    rows, columns = matrix.row.astype(np.int64), matrix.col.astype(np.int64)
    outside = np.abs(rows - columns) >= theiler
    rows, columns = rows[outside], columns[outside]
    if not rows.size:
        error_msg = f'No pair of points recurs within radius {chosen} outside the Theiler window {theiler}.'
        raise AnalysisError(error_msg)
    diagonals = _run_lengths(columns - rows, rows)
    verticals = _run_lengths(columns, rows)
    long_diagonals = diagonals[diagonals >= min_line]
    long_verticals = verticals[verticals >= min_line]
    _, counts = np.unique(long_diagonals, return_counts=True)
    shares = counts / counts.sum() if counts.size else counts
    size = array.shape[0]
    allowed = size * size - size - 2 * sum(size - offset for offset in range(1, theiler))
    return Recurrence(
        rate=float(rows.size / max(allowed, 1)),
        determinism=_line_share(diagonals, min_line, rows.size),
        laminarity=_line_share(verticals, min_line, rows.size),
        mean_diagonal=float(long_diagonals.mean()) if long_diagonals.size else 0.0,
        longest_diagonal=int(diagonals.max()),
        trapping_time=float(long_verticals.mean()) if long_verticals.size else 0.0,
        entropy=float(-np.sum(shares * np.log(shares))) if shares.size else 0.0,
        radius=chosen,
    )
