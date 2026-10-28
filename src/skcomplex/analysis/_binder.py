"""Locate transitions from samples of an order parameter: Binder cumulants, their crossings, sweeps, and overlaps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt

from ..core.errors import AnalysisError
from ._base import BaseAnalysis, Source
from ._kinds import DataKind

type Array = npt.NDArray[np.float64]
KINDS = ('cumulant', 'crossings', 'sweep', 'overlap')
SETTINGS = {
    'cumulant': set(),
    'crossings': {'parameters'},
    'sweep': {'parameters', 'size', 'temperatures'},
    'overlap': set(),
}
"""Settings every kind reads, besides `kind` and `variables`."""
MIN_SIZES = 2
REPLICAS = 2


class BinderAnalysis(BaseAnalysis):
    """Transition of an order parameter from its samples, of one kind.

    - `'cumulant'`: the Binder cumulant `1 - <m^4> / (3 <m^2>^2)` of every series of samples, 2/3 deep in an ordered
      phase and 0 for Gaussian fluctuations about 0.
    - `'sweep'`: the mean absolute order parameter, the susceptibility `size (<m²> - <|m|>²)` from its fluctuations,
      divided by the temperature when `temperatures` are given, and the Binder cumulant, for every series of samples,
      one per value of `parameters`, and the peak of the susceptibility.
    - `'crossings'`: where the Binder cumulants of consecutive system sizes cross, by linear interpolation of their
      difference. Every series holds the cumulants of one size at every value of `parameters`, in increasing size.
    - `'overlap'`: the overlap `q = <s_i t_i>` of two replicas of a configuration, stacked on the first axis of the
      source, per frame for stacks of frames.

    Args:
        kind:
            The measure to compute, one of the kinds above.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series.

        parameters:
            The control parameter values, such as temperatures or couplings, one per series of a sweep and one per
            row of the cumulants of crossings.

        size:
            The system size `N` that scales the susceptibility.

        temperatures:
            The temperature at every value that divides the susceptibility, or `None` to leave it undivided.

    Attributes:
        values_:
            The series, one per column, or the two replicas for `'overlap'`.

        labels_:
            The label of every series.

        cumulant_:
            The Binder cumulant of every series.

        crossings_:
            The first crossing of the cumulants of every pair of consecutive sizes, `nan` where they do not cross.

        parameters_:
            The control parameter values of a sweep.

        order_:
            The mean absolute order parameter at every value of a sweep.

        susceptibility_:
            The susceptibility at every value of a sweep.

        peak_:
            The control parameter value of the largest susceptibility, refined by a parabola through its neighbours.

        overlap_:
            The overlap, ±1 for identical or reversed configurations and near 0 for independent ones, one per frame
            for stacks of frames.

    Examples:
        >>> from skcomplex.analysis import BinderAnalysis
        >>> BinderAnalysis().check().prepare([1.0, -1.0, 1.0]).run().cumulant_.round(4).tolist()
        [0.6667]
        >>> cumulants = [[0.6, 0.65], [0.5, 0.5], [0.2, 0.1]]
        >>> BinderAnalysis('crossings', parameters=[1.0, 2.0, 3.0]).check().prepare(cumulants).run().crossings_.tolist()
        [2.0]
        >>> BinderAnalysis('overlap').check().prepare([[1, -1, 1, 1], [1, -1, -1, 1]]).run().overlap_
        0.5
    """

    def __init__(
        self,
        kind: str = 'cumulant',
        variables: Sequence[str] | None = None,
        *,
        parameters: npt.ArrayLike | None = None,
        size: float = 1.0,
        temperatures: npt.ArrayLike | None = None,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.parameters = parameters
        self.size = size
        self.temperatures = temperatures

    def _check(self) -> Self:
        """Validate the kind, that the kind reads every changed setting, and the control parameter values."""
        if self.kind not in KINDS:
            error_msg = f'{type(self).__name__} has kind={self.kind!r}. Use one of {", ".join(KINDS)}.'
            raise AnalysisError(error_msg)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        if self.kind in {'crossings', 'sweep'} and self.parameters is None:
            error_msg = (
                f'{type(self).__name__} of kind {self.kind!r} needs the control parameter values. Set `parameters`.'
            )
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the ensemble kind for sweeps and crossings, and the field kind otherwise."""
        return DataKind.ENSEMBLE if self.kind in {'crossings', 'sweep'} else DataKind.FIELD

    def _prepare(self, source: Source) -> Self:
        """Read the two replicas of an overlap, or the series of the source as columns."""
        if self.kind == 'overlap':
            self.values_ = np.asarray(source, dtype=float)
            if self.values_.shape[0] != REPLICAS:
                error_msg = (
                    f'An overlap reads {REPLICAS} replicas stacked on the first axis, but the source has '
                    f'{self.values_.shape[0]}. Stack the two replicas with `np.stack`.'
                )
                raise AnalysisError(error_msg)
            self.labels_ = ['first', 'second']
            return self
        series = self._read_series(source, self.variables)
        self.values_, self.labels_ = np.asarray(series.values, dtype=float), series.labels
        return self

    def _run(self) -> Self:
        """Compute the cumulants, their crossings, the sweep, or the overlap."""
        if self.kind == 'cumulant':
            self.cumulant_ = self._cumulants(self.values_)
        elif self.kind == 'crossings':
            if self.values_.shape[1] < MIN_SIZES:
                error_msg = (
                    f'Crossings need at least {MIN_SIZES} sizes, not {self.values_.shape[1]}. Give one series per size.'
                )
                raise AnalysisError(error_msg)
            self.crossings_ = self._crossings(np.asarray(self.parameters, dtype=float), self.values_)
        elif self.kind == 'sweep':
            absolute = np.abs(self.values_)
            self.parameters_ = np.asarray(self.parameters, dtype=float)
            self.order_ = np.asarray([column.mean() for column in absolute.T])
            self.susceptibility_ = self.size * np.asarray(
                [np.mean(column**2) - column.mean() ** 2 for column in absolute.T],
            )
            if self.temperatures is not None:
                self.susceptibility_ = self.susceptibility_ / np.asarray(self.temperatures, dtype=float)
            self.cumulant_ = self._cumulants(absolute)
            self.peak_ = self._refined_peak(self.parameters_, self.susceptibility_)
        else:
            product = self.values_[0] * self.values_[1]
            self.overlap_ = (
                float(product.mean()) if product.ndim == 1 else product.reshape(product.shape[0], -1).mean(axis=1)
            )
        return self

    @staticmethod
    def _cumulants(samples: Array) -> Array:
        """Return the Binder cumulant `1 - <m^4> / (3 <m^2>^2)` of every column of samples."""
        return np.asarray([1 - np.mean(column**4) / (3 * np.mean(column**2) ** 2) for column in samples.T])

    @staticmethod
    def _refined_peak(grid: Array, values: Array) -> float:
        """Return the abscissa of the vertex of the parabola through the largest value and its neighbours."""
        index = int(np.argmax(values))
        if index in (0, values.size - 1):
            return float(grid[index])
        coefficients = np.polyfit(grid[index - 1 : index + 2], values[index - 1 : index + 2], 2)
        if coefficients[0] >= 0:
            return float(grid[index])
        return float(-coefficients[1] / (2 * coefficients[0]))

    @staticmethod
    def _crossings(grid: Array, cumulants: Array) -> Array:
        """Return the first crossing of the cumulants of every pair of consecutive columns, `nan` without one."""
        crossings = []
        for small, large in itertools.pairwise(cumulants.T):
            difference = large - small
            changes = np.flatnonzero(np.sign(difference[:-1]) * np.sign(difference[1:]) <= 0)
            if not changes.size:
                crossings.append(np.nan)
                continue
            index = int(changes[0])
            left, right = difference[index], difference[index + 1]
            share = left / (left - right) if left != right else 0.0
            crossings.append(grid[index] + share * (grid[index + 1] - grid[index]))
        return np.asarray(crossings, dtype=float)
