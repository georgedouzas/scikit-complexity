"""Embed series in delay coordinates and choose the delay and the dimension from the data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from ..base import AnalysisError
from ._information import mutual_information, symbolize
from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]


@analysis(DataKind.SERIES)
def delay_embedding(values: npt.ArrayLike, dimension: int, delay: int = 1) -> Array:
    """Return the delay vectors `(x_t, x_{t+delay}, ..., x_{t+(dimension-1) delay})` of a series.

    Args:
        values:
            The series.

        dimension:
            The number of coordinates.

        delay:
            The delay in steps.

    Returns:
        points:
            One row per delay vector.

    Raises:
        AnalysisError:
            The series is too short for one vector.

    Examples:
        >>> from skcomplex.core.analysis import delay_embedding
        >>> delay_embedding([0, 1, 2, 3, 4], 2, 2).tolist()
        [[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]]
    """
    array = np.asarray(values, dtype=float)
    size = array.size - (dimension - 1) * delay
    if size < 1:
        error_msg = f'A series of {array.size} values has no delay vector of dimension {dimension} and delay {delay}.'
        raise AnalysisError(error_msg)
    return np.column_stack([array[index * delay : index * delay + size] for index in range(dimension)])


@analysis(DataKind.SERIES)
def auto_mutual_information(values: npt.ArrayLike, max_delay: int = 100, bins: int = 16) -> Array:
    """Return the mutual information in bits between a series and itself delayed by 0 to `max_delay` steps.

    Args:
        values:
            The series.

        max_delay:
            The largest delay in steps.

        bins:
            The number of equally populated bins the values are symbolized into.

    Returns:
        information:
            The mutual information at every delay.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import auto_mutual_information
        >>> information = auto_mutual_information(np.random.default_rng(0).standard_normal(10_000), 2, bins=4)
        >>> round(float(information[0]), 2), bool(information[1] < 0.01)
        (2.0, True)
    """
    symbols = symbolize(values, 'quantiles', bins)
    return np.asarray(
        [mutual_information(symbols[: symbols.size - delay], symbols[delay:]) for delay in range(max_delay + 1)],
    )


def _first_minimum(values: Array) -> int:
    """Return the index of the first local minimum, or of the global minimum without one."""
    falls = np.flatnonzero((values[1:-1] <= values[:-2]) & (values[1:-1] < values[2:]))
    return int(falls[0] + 1) if falls.size else int(values.argmin())


@analysis(DataKind.SERIES)
def false_nearest_neighbors(
    values: npt.ArrayLike,
    delay: int = 1,
    max_dimension: int = 10,
    *,
    ratio: float = 10.0,
    escape: float = 2.0,
) -> Array:
    """Return Kennel's fraction of false nearest neighbours of the delay embeddings of dimensions 1 to `max_dimension`.

    A nearest neighbour in dimension `d` is false when the next coordinate moves it away by more than `ratio` times
    their distance, or by more than `escape` times the standard deviation of the series.

    Args:
        values:
            The series.

        delay:
            The delay in steps.

        max_dimension:
            The largest dimension.

        ratio:
            The threshold on the relative growth of the distance.

        escape:
            The threshold on the distance in the next dimension, in standard deviations.

    Returns:
        fractions:
            The fraction of false neighbours in every dimension from 1.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import false_nearest_neighbors
        >>> x = np.sin(np.arange(3000) * 0.05)
        >>> false_nearest_neighbors(x, 30, 3).round(2).tolist()[1:]
        [0.0, 0.0]
    """
    array = np.asarray(values, dtype=float)
    spread = array.std()
    fractions = []
    for dimension in range(1, max_dimension + 1):
        points = delay_embedding(array, dimension, delay)
        size = array.size - dimension * delay
        distances, neighbours = cKDTree(points[:size]).query(points[:size], k=2)
        distance, neighbour = np.maximum(distances[:, 1], 1e-300), neighbours[:, 1]
        step = np.abs(array[np.arange(size) + dimension * delay] - array[neighbour + dimension * delay])
        false = (step / distance > ratio) | (np.sqrt(distance**2 + step**2) / spread > escape)
        fractions.append(false.mean())
    return np.asarray(fractions)


@dataclass(frozen=True)
class Embedding:
    """Delay embedding of a series with the delay and the dimension chosen from the data.

    Args:
        points:
            One row per delay vector.

        delay:
            The delay in steps, at the first minimum of the auto mutual information.

        dimension:
            The first dimension whose fraction of false nearest neighbours lies below the threshold.

        mutual_information:
            The auto mutual information at every delay from 0.

        false_neighbors:
            The fraction of false nearest neighbours in every dimension from 1.
    """

    points: Array
    delay: int
    dimension: int
    mutual_information: Array
    false_neighbors: Array


@analysis(DataKind.SERIES)
def embed(
    values: npt.ArrayLike,
    *,
    delay: int | None = None,
    dimension: int | None = None,
    threshold: float = 0.01,
    max_delay: int = 100,
    max_dimension: int = 10,
    bins: int = 16,
) -> Embedding:
    """Return the delay embedding of a series, with the delay and the dimension chosen from the data unless given.

    Args:
        values:
            The series.

        delay:
            The delay in steps, or `None` for the first minimum of the auto mutual information.

        dimension:
            The dimension, or `None` for the first one with a fraction of false nearest neighbours below `threshold`.

        threshold:
            The fraction of false nearest neighbours accepted.

        max_delay:
            The largest delay tried.

        max_dimension:
            The largest dimension tried.

        bins:
            The number of bins of the auto mutual information.

    Returns:
        embedding:
            The points, the delay, the dimension, and the curves they were chosen from.

    Raises:
        AnalysisError:
            No dimension up to `max_dimension` reaches the threshold.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import embed
        >>> x = np.sin(np.arange(5000) * 0.05)
        >>> embedding = embed(x, delay=31)
        >>> embedding.delay, embedding.dimension
        (31, 2)
    """
    array = np.asarray(values, dtype=float)
    information = auto_mutual_information(array, max_delay, bins)
    chosen_delay = _first_minimum(information) if delay is None else delay
    fractions = false_nearest_neighbors(array, chosen_delay, max_dimension)
    if dimension is None:
        below = np.flatnonzero(fractions < threshold)
        if not below.size:
            error_msg = f'No dimension up to {max_dimension} has fewer than {threshold} false neighbours.'
            raise AnalysisError(error_msg)
        dimension = int(below[0] + 1)
    return Embedding(delay_embedding(array, dimension, chosen_delay), chosen_delay, dimension, information, fractions)
