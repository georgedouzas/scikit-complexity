"""Embed series in delay coordinates and choose the delay and the dimension from the data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from ..errors import AnalysisError
from ..types import Array
from ._base import BaseAnalysis, Source
from ._information import InformationAnalysis
from ._kinds import DataKind

KINDS = ('delay', 'false-nearest-neighbors')
SETTINGS = {
    'delay': {'delay', 'dimension', 'threshold', 'max_delay', 'max_dimension', 'bins'},
    'false-nearest-neighbors': {'delay', 'max_delay', 'max_dimension', 'bins', 'ratio', 'escape'},
}
"""Settings every kind reads, besides `kind` and `variables`."""


class EmbeddingAnalysis(BaseAnalysis):
    """Delay embedding of one series, with the delay and the dimension chosen from the data unless given, of one kind.

    The kinds read one series:

    - `'delay'`: the delay vectors `(x_t, x_{t+delay}, ..., x_{t+(dimension-1) delay})`. Without a delay, the delay is
      the first minimum of the auto mutual information. Without a dimension, the dimension is the first one whose
      fraction of false nearest neighbours lies below `threshold`.
    - `'false-nearest-neighbors'`: Kennel's fraction of false nearest neighbours of the embeddings of dimensions 1 to
      `max_dimension`. A nearest neighbour in dimension `d` is false when the next coordinate moves it away by more
      than `ratio` times their distance, or by more than `escape` times the standard deviation of the series.

    Args:
        kind:
            The embedding measure to compute, one of the kinds above.

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

    Attributes:
        values_:
            The series, as one column.

        labels_:
            The label of the series.

        delay_:
            The delay in steps.

        dimension_:
            The dimension of `'delay'`.

        points_:
            One row per delay vector of `'delay'`.

        mutual_information_:
            The auto mutual information at every delay from 0, or `None` when the delay is given.

        false_neighbors_:
            The fraction of false nearest neighbours in every dimension from 1, or `None` when `'delay'` is given the
            dimension.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import EmbeddingAnalysis
        >>> EmbeddingAnalysis(delay=2, dimension=2).check().prepare([0, 1, 2, 3, 4]).run().points_.tolist()
        [[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]]
        >>> embedding = EmbeddingAnalysis(delay=31).check().prepare(np.sin(np.arange(5000) * 0.05)).run()
        >>> embedding.delay_, embedding.dimension_
        (31, 2)
    """

    def __init__(
        self,
        kind: str = 'delay',
        variables: Sequence[str] | None = None,
        *,
        delay: int | None = None,
        dimension: int | None = None,
        threshold: float = 0.01,
        max_delay: int = 100,
        max_dimension: int = 10,
        bins: int = 16,
        ratio: float = 10.0,
        escape: float = 2.0,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.delay = delay
        self.dimension = dimension
        self.threshold = threshold
        self.max_delay = max_delay
        self.max_dimension = max_dimension
        self.bins = bins
        self.ratio = ratio
        self.escape = escape

    def _check(self) -> Self:
        """Validate the kind, the delays and dimensions, and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        sizes = [size for size in (self.delay, self.dimension) if size is not None]
        if any(size < 1 for size in [*sizes, self.max_delay, self.max_dimension, self.bins]):
            error_msg = (
                f'{type(self).__name__} has delay={self.delay!r}, dimension={self.dimension!r}, '
                f'max_delay={self.max_delay!r}, max_dimension={self.max_dimension!r}, and bins={self.bins!r}. '
                'Use 1 or more.'
            )
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the series kind."""
        return DataKind.SERIES

    def _prepare(self, source: Source) -> Self:
        """Read the one series to embed."""
        series = self._read_series(source, self.variables)
        if series.values.shape[1] != 1:
            error_msg = (
                f'{type(self).__name__} embeds one series, but the source has {series.values.shape[1]}: '
                f'{", ".join(series.labels)}. Pick one with `variables`.'
            )
            raise AnalysisError(error_msg)
        self.values_ = series.values
        self.labels_ = series.labels
        return self

    def _run(self, inputs: object = None) -> Self:
        """Choose the delay and the dimension unless given, and compute the embedding or the false neighbours."""
        array = self.values_[:, 0]
        self.mutual_information_ = None
        self.false_neighbors_ = None
        self.delay_ = self.delay
        if self.delay is None:
            self.mutual_information_ = self._auto_mutual_information(array, self.max_delay, self.bins)
            self.delay_ = self._first_minimum(self.mutual_information_)
        if self.kind == 'false-nearest-neighbors':
            self.false_neighbors_ = self._false_nearest_neighbors(
                array,
                self.delay_,
                self.max_dimension,
                ratio=self.ratio,
                escape=self.escape,
            )
            return self
        self.dimension_ = self.dimension
        if self.dimension is None:
            self.false_neighbors_ = self._false_nearest_neighbors(array, self.delay_, self.max_dimension)
            below = np.flatnonzero(self.false_neighbors_ < self.threshold)
            if not below.size:
                error_msg = (
                    f'No dimension up to {self.max_dimension} has fewer than {self.threshold} false neighbours. '
                    'Raise max_dimension or threshold.'
                )
                raise AnalysisError(error_msg)
            self.dimension_ = int(below[0] + 1)
        self.points_ = self._delay_embedding(array, self.dimension_, self.delay_)
        return self

    @staticmethod
    def _first_minimum(values: Array) -> int:
        """Return the index of the first local minimum, or of the global minimum without one."""
        falls = np.flatnonzero((values[1:-1] <= values[:-2]) & (values[1:-1] < values[2:]))
        return int(falls[0] + 1) if falls.size else int(values.argmin())

    @staticmethod
    def _delay_embedding(values: npt.ArrayLike, dimension: int, delay: int = 1) -> Array:
        """Return the delay vectors `(x_t, x_{t+delay}, ..., x_{t+(dimension-1) delay})` of a series, one per row."""
        array = np.asarray(values, dtype=float)
        size = array.size - (dimension - 1) * delay
        if size < 1:
            error_msg = (
                f'A series of {array.size} values has no delay vector of dimension {dimension} and delay {delay}. '
                'Use a longer series, or a smaller dimension or delay.'
            )
            raise AnalysisError(error_msg)
        return np.column_stack([array[index * delay : index * delay + size] for index in range(dimension)])

    @staticmethod
    def _false_nearest_neighbors(
        values: npt.ArrayLike,
        delay: int = 1,
        max_dimension: int = 10,
        *,
        ratio: float = 10.0,
        escape: float = 2.0,
    ) -> Array:
        """Return Kennel's share of false nearest neighbours of the delay embeddings of dimension 1 to the largest."""
        array = np.asarray(values, dtype=float)
        spread = array.std()
        fractions = []
        for dimension in range(1, max_dimension + 1):
            points = EmbeddingAnalysis._delay_embedding(array, dimension, delay)
            size = array.size - dimension * delay
            distances, neighbours = cKDTree(points[:size]).query(points[:size], k=2)
            distance, neighbour = np.maximum(distances[:, 1], 1e-300), neighbours[:, 1]
            step = np.abs(array[np.arange(size) + dimension * delay] - array[neighbour + dimension * delay])
            false = (step / distance > ratio) | (np.sqrt(distance**2 + step**2) / spread > escape)
            fractions.append(false.mean())
        return np.asarray(fractions)

    @staticmethod
    def _auto_mutual_information(values: npt.ArrayLike, max_delay: int = 100, bins: int = 16) -> Array:
        """Return the mutual information in bits of a series in `bins` equally populated bins with itself delayed."""
        symbols = InformationAnalysis._symbolize(values, 'quantiles', bins)
        return InformationAnalysis._delayed_information(symbols, max_delay, base=2.0)
