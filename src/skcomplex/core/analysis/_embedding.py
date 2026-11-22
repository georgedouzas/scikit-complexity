"""Embed series in delay coordinates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from ..enums import DataKind
from ..errors import AnalysisError
from ..types import Array
from ._analysis_tags import AnalysisTags
from ._base import _check_choice, _check_settings, _read_series
from ._information import _delayed_information, _symbolize
from .types import DataSource

SETTINGS = {
    'delay': {'delay', 'dimension', 'threshold', 'max_delay', 'max_dimension', 'bins'},
    'false-nearest-neighbors': {'delay', 'max_delay', 'max_dimension', 'bins', 'ratio', 'escape'},
}
KINDS = ('delay', 'false-nearest-neighbors')
TAGS = AnalysisTags(DataKind.SERIES)


@dataclass(frozen=True)
class EmbeddingResult:
    """A result of `embedding`, the delayed copies of a series.

    It holds the delay, the dimension, the points made of delayed copies of the series, and the measures that chose
    them, the auto mutual information and the share of false neighbours. It reports the rebuilt orbit of a heart beat,
    of a river flow, or of a price.

    Read more in the user guide, under Analyse.

    Args:
        values:
            The series, as one column.

        labels:
            The label of the series.

        delay:
            The delay in steps.

        dimension:
            The dimension of `'delay'`, or `None` for `'false-nearest-neighbors'`.

        points:
            One row per delay vector of `'delay'`, or `None` for `'false-nearest-neighbors'`.

        mutual_information:
            The auto mutual information at every delay from 0, or `None` when the delay is given.

        false_neighbors:
            The fraction of false nearest neighbours in every dimension from 1, or `None` when `'delay'` is given the
            dimension.

    See Also:
        embedding: Returns this result.

    Examples:
        >>> from skcomplex.core.analysis import embedding
        >>> embedding([0, 1, 2, 3, 4], delay=2, dimension=2).points.tolist()
        [[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]]
    """

    values: Array
    labels: list[str]
    delay: int
    dimension: int | None = None
    points: Array | None = None
    mutual_information: Array | None = None
    false_neighbors: Array | None = None


def embedding(
    source: DataSource,
    *,
    kind: str = 'delay',
    variables: Sequence[str] | None = None,
    delay: int | None = None,
    dimension: int | None = None,
    threshold: float = 0.01,
    max_delay: int = 100,
    max_dimension: int = 10,
    bins: int = 16,
    ratio: float = 10.0,
    escape: float = 2.0,
) -> EmbeddingResult:
    """Return the points made of delayed copies of one series.

    It builds the points `(x_t, x_{t+delay}, ..., x_{t+(dimension-1) delay})`, which rebuild the shape of the orbit of a
    system from one measured series, such as `[[0, 2], [1, 3], [2, 4]]` for `0, 1, 2, 3, 4` with delay 2. Without a
    delay it takes the first minimum of the auto mutual information, and without a dimension the first one whose share
    of false neighbours lies below `threshold`. It rebuilds the orbit of a heart rate, of a river flow, and of a stock
    price.

    Read more in the user guide, under Analyse.

    Args:
        source:
            The trajectory, or an array that holds one series.

        kind:
            `'delay'` gives the delayed points. `'false-nearest-neighbors'` gives the share of false neighbours of
            Kennel for dimensions 1 to `max_dimension`, a neighbour in dimension `d` being false when the next
            coordinate moves it away by more than `ratio` times their distance or `escape` times the spread of the
            series.

        variables:
            The one variable of a trajectory to read, or `None` for the only scalar series.

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
            The number of equally populated bins of the auto mutual information.

        ratio:
            The threshold on the relative growth of the distance of a false neighbour.

        escape:
            The threshold on the distance of a false neighbour in the next dimension, in standard deviations.

    Returns:
        result:
            The series, the delay, and the embedding or the fraction of false neighbours.

    Raises:
        AnalysisError:
            The kind is unknown, a setting the kind does not read is changed, a size is below 1, the source holds
            several series, or no dimension has few enough false neighbours.

    See Also:
        recurrence: Finds when an orbit returns near itself.
        lyapunov: Measures how fast nearby runs drift apart.

    Examples:
        >>> from skcomplex.core.analysis import embedding
        >>> embedding([0, 1, 2, 3, 4], delay=2, dimension=2).points.tolist()
        [[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]]
    """
    _check_choice('embedding', 'kind', kind, KINDS)
    _check_settings(
        embedding,
        'kind',
        SETTINGS[kind] | {'variables'},
        kind=kind,
        variables=variables,
        delay=delay,
        dimension=dimension,
        threshold=threshold,
        max_delay=max_delay,
        max_dimension=max_dimension,
        bins=bins,
        ratio=ratio,
        escape=escape,
    )
    sizes = [size for size in (delay, dimension) if size is not None]
    if any(size < 1 for size in [*sizes, max_delay, max_dimension, bins]):
        error_msg = (
            f'embedding has delay={delay!r}, dimension={dimension!r}, max_delay={max_delay!r}, '
            f'max_dimension={max_dimension!r}, and bins={bins!r}. Use 1 or more.'
        )
        raise AnalysisError(error_msg)
    series = _read_series(source, variables)
    if series.values.shape[1] != 1:
        error_msg = (
            f'embedding embeds one series, but the source has {series.values.shape[1]}: {", ".join(series.labels)}. '
            'Pick one with `variables`.'
        )
        raise AnalysisError(error_msg)
    array = series.values[:, 0]
    mutual_information = None
    if delay is None:
        mutual_information = _auto_mutual_information(array, max_delay, bins)
        delay = _first_minimum(mutual_information)
    false_neighbors: Array | None = None
    if kind == 'false-nearest-neighbors':
        false_neighbors = _false_nearest_neighbors(array, delay, max_dimension, ratio=ratio, escape=escape)
        return EmbeddingResult(
            series.values,
            series.labels,
            delay,
            mutual_information=mutual_information,
            false_neighbors=false_neighbors,
        )
    if dimension is None:
        false_neighbors = _false_nearest_neighbors(array, delay, max_dimension)
        below = np.flatnonzero(false_neighbors < threshold)
        if not below.size:
            error_msg = (
                f'No dimension up to {max_dimension} has fewer than {threshold} false neighbours. Raise '
                'max_dimension or threshold.'
            )
            raise AnalysisError(error_msg)
        dimension = int(below[0] + 1)
    return EmbeddingResult(
        series.values,
        series.labels,
        delay,
        dimension,
        _delay_embedding(array, dimension, delay),
        mutual_information,
        false_neighbors,
    )


def _first_minimum(values: Array) -> int:
    """Return the index of the first local minimum.

    Without one, the index of the global minimum is returned.
    """
    falls = np.flatnonzero((values[1:-1] <= values[:-2]) & (values[1:-1] < values[2:]))
    return int(falls[0] + 1) if falls.size else int(values.argmin())


def _delay_embedding(values: npt.ArrayLike, dimension: int, delay: int = 1) -> Array:
    """Return the delay vectors of a series, one per row.

    The vectors are `(x_t, x_{t+delay}, ..., x_{t+(dimension-1) delay})`.
    """
    array = np.asarray(values, dtype=float)
    size = array.size - (dimension - 1) * delay
    if size < 1:
        error_msg = (
            f'A series of {array.size} values has no delay vector of dimension {dimension} and delay {delay}. '
            'Use a longer series, or a smaller dimension or delay.'
        )
        raise AnalysisError(error_msg)
    return np.column_stack([array[index * delay : index * delay + size] for index in range(dimension)])


def _false_nearest_neighbors(
    values: npt.ArrayLike,
    delay: int = 1,
    max_dimension: int = 10,
    *,
    ratio: float = 10.0,
    escape: float = 2.0,
) -> Array:
    """Return Kennel's share of false nearest neighbours of the delay embeddings.

    The dimensions run from 1 to the largest.
    """
    array = np.asarray(values, dtype=float)
    spread = array.std()
    fractions = []
    for dimension in range(1, max_dimension + 1):
        points = _delay_embedding(array, dimension, delay)
        size = array.size - dimension * delay
        distances, neighbours = cKDTree(points[:size]).query(points[:size], k=2)
        distance, neighbour = np.maximum(distances[:, 1], 1e-300), neighbours[:, 1]
        step = np.abs(array[np.arange(size) + dimension * delay] - array[neighbour + dimension * delay])
        false = (step / distance > ratio) | (np.sqrt(distance**2 + step**2) / spread > escape)
        fractions.append(false.mean())
    return np.asarray(fractions)


def _auto_mutual_information(values: npt.ArrayLike, max_delay: int = 100, bins: int = 16) -> Array:
    """Return the mutual information in bits of a series with itself delayed.

    The series is binned in `bins` equally populated bins.
    """
    symbols = _symbolize(values, 'quantiles', bins)
    return _delayed_information(symbols, max_delay, base=2.0)
