"""Measure avalanches: their sizes, durations, and shapes, crackling-noise scaling, and branching ratios."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt

from ..core.errors import AnalysisError
from ._base import BaseAnalysis, Source
from ._kinds import DataKind
from ._lazy import module

type Array = npt.NDArray[np.float64]
KINDS = ('sizes', 'crackling', 'branching')
SETTINGS = {
    'sizes': {'threshold'},
    'crackling': {'threshold', 'durations', 'min_count', 'min_duration', 'size_exponent', 'duration_exponent'},
    'branching': {'max_step', 'dt'},
}
"""Settings every kind reads, besides `kind` and `variables`."""
MIN_POINTS = 2
SHAPE_GRID = 50


@runtime_checkable
class _Relaxation(Protocol):
    """Record of the relaxation of a lattice within one step, with its size and its duration."""

    size: int
    duration: int
    area: int


def _records(source: object) -> list[_Relaxation]:
    """Return the relaxation records of a source, or nothing when it holds activity values."""
    if isinstance(source, np.ndarray) or not isinstance(source, Sequence):
        return []
    records = [item for item in source if isinstance(item, _Relaxation)]
    return records if records and len(records) == len(source) else []


def _mean_shape(shapes: list[Array]) -> Array:
    """Return the mean profile of avalanches of one duration, scaled by the mean height, on a grid of 0 to 1."""
    profiles = np.asarray(shapes, dtype=float)
    grid = np.linspace(0, 1, SHAPE_GRID)
    steps = np.linspace(0, 1, profiles.shape[1])
    mean = profiles.mean(axis=0)
    return np.interp(grid, steps, mean / mean.mean())


class AvalancheAnalysis(BaseAnalysis):
    """Avalanches of an activity series or of the relaxations of a lattice, of one kind.

    An avalanche of an activity series is a run of steps whose activity exceeds `threshold`, and its profile is the
    activity above the threshold. The relaxation records of a trajectory, such as the events of a sandpile, are
    avalanches already.

    - `'sizes'`: the size, the duration, and the profile of every avalanche.
    - `'crackling'`: the avalanches, the exponent `gamma` of the mean size against the duration, `<S>(T) ~ T^gamma`,
      its prediction `(tau_T - 1) / (tau_S - 1)` from the exponents of the size and duration laws, and the collapse of
      the mean shapes of the chosen `durations`.
    - `'branching'`: the branching ratio of the activity by the multistep regression estimator of `mrestimator`, which
      fits `r_k ∝ m^k` to the regression slopes `r_k` of the activity `k` steps apart, and stays unbiased when only a
      fraction of the units is observed. Every series is one trial.

    Args:
        kind:
            The measure to compute, one of the kinds above.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series.

        threshold:
            The activity an avalanche exceeds at every step, subtracted from its profile.

        durations:
            The durations whose mean shapes are compared.

        min_count:
            The fewest avalanches of a duration used in the fit of the mean size.

        min_duration:
            The shortest duration used in the fit of the mean size.

        size_exponent:
            The exponent `tau_S` of the power law of the sizes, or `None` to leave out the prediction.

        duration_exponent:
            The exponent `tau_T` of the power law of the durations, or `None` to leave out the prediction.

        max_step:
            The largest step `k` of the regression of the branching ratio.

        dt:
            The duration of a step.

    Attributes:
        activity_:
            The activity per step, one column per series, or `None` for relaxation records.

        labels_:
            The label of every series.

        sizes_:
            The total activity of every avalanche.

        durations_:
            The number of steps every avalanche lasts.

        shapes_:
            The activity profile of every avalanche of an activity series, empty for relaxation records.

        exponent_:
            The exponent `gamma` of the mean size against the duration.

        predicted_:
            The exponent `(tau_T - 1) / (tau_S - 1)`, or `nan` without both exponents.

        shape_spread_:
            The mean over the inner 80 percent of the scaled time of the relative difference between the largest and
            the smallest mean shape of the chosen durations, `nan` with fewer than 2 of them.

        mean_shapes_:
            The mean shape of every chosen duration, scaled to unit mean height, on a grid of 0 to 1.

        branching_ratio_:
            The branching ratio `m`, 1 at criticality.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import AvalancheAnalysis
        >>> found = AvalancheAnalysis().check().prepare([0, 1, 2, 1, 0, 3, 0]).run()
        >>> found.sizes_.tolist(), found.durations_.tolist()
        ([4.0, 3.0], [3.0, 1.0])
        >>> rng = np.random.default_rng(0)
        >>> activity = np.zeros(20_000)
        >>> activity[0] = 100
        >>> for t in range(1, activity.size):
        ...     activity[t] = rng.poisson(0.9 * activity[t - 1] + 10)
        >>> round(AvalancheAnalysis('branching').check().prepare(activity).run().branching_ratio_, 1)
        0.9
    """

    def __init__(
        self,
        kind: str = 'sizes',
        variables: Sequence[str] | None = None,
        *,
        threshold: float = 0.0,
        durations: Sequence[int] = (),
        min_count: int = 10,
        min_duration: int = 1,
        size_exponent: float | None = None,
        duration_exponent: float | None = None,
        max_step: int = 40,
        dt: float = 1.0,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.threshold = threshold
        self.durations = durations
        self.min_count = min_count
        self.min_duration = min_duration
        self.size_exponent = size_exponent
        self.duration_exponent = duration_exponent
        self.max_step = max_step
        self.dt = dt

    def _check(self) -> Self:
        """Validate the kind, that the kind reads every changed setting, and the counts and steps."""
        if self.kind not in KINDS:
            error_msg = f'{type(self).__name__} has kind={self.kind!r}. Use one of {", ".join(KINDS)}.'
            raise AnalysisError(error_msg)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        if self.min_count < 1 or self.max_step < 1 or self.dt <= 0:
            error_msg = (
                f'{type(self).__name__} has min_count={self.min_count!r}, max_step={self.max_step!r}, and '
                f'dt={self.dt!r}. Use positive values.'
            )
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the series kind."""
        return DataKind.SERIES

    def _find(self) -> None:
        """Find the runs of the activity above the threshold, with their sizes, durations, and profiles."""
        if self.activity_.shape[1] != 1:
            error_msg = (
                f'Avalanches are found in one activity series, but the source has {self.activity_.shape[1]}. Choose '
                'one with `variables`.'
            )
            raise AnalysisError(error_msg)
        activity = self.activity_[:, 0] - self.threshold
        active = np.concatenate([[False], activity > 0, [False]]).astype(int)
        starts, ends = np.flatnonzero(np.diff(active) == 1), np.flatnonzero(np.diff(active) == -1)
        self.shapes_ = [activity[start:end] for start, end in zip(starts, ends, strict=True)]
        self.sizes_ = np.asarray([shape.sum() for shape in self.shapes_], dtype=float)
        self.durations_ = np.asarray([shape.size for shape in self.shapes_], dtype=float)

    def _prepare(self, source: Source | Sequence[_Relaxation]) -> Self:
        """Read the relaxation records, or the activity of the source and its avalanches."""
        records = _records(source)
        if records:
            if self.kind == 'branching':
                error_msg = 'The branching ratio is measured on an activity series, not on relaxation records.'
                raise AnalysisError(error_msg)
            self.activity_, self.labels_ = None, ['relaxations']
            self.sizes_ = np.asarray([record.size for record in records], dtype=np.float64)
            self.durations_ = np.asarray([record.duration for record in records], dtype=np.float64)
            self.shapes_ = []
            return self
        series = self._read_series(source, self.variables)
        self.activity_, self.labels_ = np.asarray(series.values, dtype=float), series.labels
        if self.kind != 'branching':
            self._find()
        return self

    def _run_crackling(self) -> None:
        """Fit the mean size against the duration, predict its exponent, and compare the mean shapes."""
        distinct, counts = np.unique(self.durations_, return_counts=True)
        kept = distinct[(counts >= self.min_count) & (distinct >= max(self.min_duration, 1))]
        if kept.size < MIN_POINTS:
            error_msg = (
                f'Fewer than {MIN_POINTS} durations have {self.min_count} avalanches. Give more avalanches or lower '
                '`min_count`.'
            )
            raise AnalysisError(error_msg)
        means = np.asarray([self.sizes_[self.durations_ == duration].mean() for duration in kept])
        self.exponent_ = float(np.polyfit(np.log(kept), np.log(means), 1)[0])
        self.predicted_ = float('nan')
        if self.size_exponent is not None and self.duration_exponent is not None:
            self.predicted_ = float((self.duration_exponent - 1) / (self.size_exponent - 1))
        self.mean_shapes_ = {
            int(duration): _mean_shape([shape for shape in self.shapes_ if shape.size == duration])
            for duration in self.durations
            if any(shape.size == duration for shape in self.shapes_)
        }
        self.shape_spread_ = float('nan')
        if len(self.mean_shapes_) >= MIN_POINTS:
            stacked = np.asarray(list(self.mean_shapes_.values()))
            inner = slice(SHAPE_GRID // 10, SHAPE_GRID - SHAPE_GRID // 10)
            spread = (stacked.max(axis=0) - stacked.min(axis=0))[inner] / stacked.mean(axis=0)[inner]
            self.shape_spread_ = float(np.mean(spread))

    def _run(self) -> Self:
        """Compute the crackling-noise scaling or the branching ratio, the avalanches being found already."""
        if self.kind == 'crackling':
            self._run_crackling()
        elif self.kind == 'branching':
            mrestimator = module('mrestimator')
            coefficients = mrestimator.coefficients(
                self.activity_.T,
                method='ts',
                steps=(1, self.max_step),
                dt=self.dt,
                dtunit='step',
                numboot=0,
            )
            result = mrestimator.fit(coefficients, fitfunc=mrestimator.f_exponential_offset, numboot=0)
            self.branching_ratio_ = float(result.mre)
        return self
