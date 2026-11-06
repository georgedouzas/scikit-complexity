"""Detect events in series, count them in windows, time laminar phases, find excursions, and estimate branching."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..specs import CascadeRecordSpec
from ..types import Array, DataKind
from ._base import BaseAnalysis
from .types import Source

KINDS = ('detect', 'counts', 'laminar-durations', 'excursions', 'branching')
SETTINGS = {
    'detect': {'threshold'},
    'counts': {'window', 'start', 'end'},
    'laminar-durations': {'threshold'},
    'excursions': {'threshold'},
    'branching': {'max_step', 'dt'},
}
"""Settings every kind reads, besides `kind` and `variables`."""


class EventAnalysis(BaseAnalysis):
    """Events of one series or of event times, of one kind.

    The kinds read one series or the times of events:

    - `'detect'`: the events of a series, every maximal run of consecutive values above `threshold`, with their
      starts, sizes, durations, peaks, and the waiting times between them. For avalanches recorded as topplings per
      drive step, an event is an avalanche and its size is the total number of topplings.
    - `'counts'`: the number of events in consecutive windows of length `window`, from the times of the events, or
      from the jump times of a discrete state of a trajectory.
    - `'laminar-durations'`: the durations of the laminar phases of a series, the complete runs of values within
      `threshold` of 0 between two bursts, such as the distance from an invariant manifold of on-off intermittency.
    - `'excursions'`: the excursions of a series above `threshold`, every maximal run of values above it, with the
      profile of every excursion, its values minus the threshold, its size, the sum of the profile, and its duration,
      the number of its samples. A sequence of records with a `size` and a `duration`, such as the cascade records
      of a lattice, holds excursions already.
    - `'branching'`: the branching ratio of the series by the multistep regression estimator of `mrestimator`, which
      fits `r_k ∝ m^k` to the regression slopes `r_k` of the values `k` steps apart, and stays unbiased when only a
      fraction of the units is observed. Every series is one trial.

    Times are the times of a trajectory, and the step indices of an array.

    Args:
        kind:
            The events to measure, one of the kinds above.

        variables:
            The one variable of a trajectory to read, or `None` for the only scalar series or discrete state.

        threshold:
            The value an event must exceed, or the largest absolute value of a laminar phase.

        window:
            The length of every window of `'counts'`.

        start:
            The start of the first window.

        end:
            The end of the last full window, or `None` for the last event.

        max_step:
            The largest step `k` of the regression of the branching ratio.

        dt:
            The duration of a step of the branching ratio.

    Attributes:
        times_:
            The time of every value, or of every event for `'counts'`.

        values_:
            The series, as one column, or one column per trial for `'branching'`, or `None` for records.

        labels_:
            The label of the series or the state.

        starts_:
            The time at which every event starts.

        sizes_:
            The sum of the values of every event.

        durations_:
            The time every event lasts for `'detect'`, every laminar phase lasts for `'laminar-durations'`, or the
            number of samples of every excursion for `'excursions'`.

        shapes_:
            The profile of every excursion of a series, empty for records.

        peaks_:
            The largest value of every event.

        waiting_times_:
            The time between the end of every event and the start of the next.

        counts_:
            The number of events in every window.

        branching_ratio_:
            The branching ratio `m`, 1 at criticality.

    Examples:
        >>> from skcomplex.core.analysis import EventAnalysis
        >>> events = EventAnalysis().prepare([0, 3, 1, 0, 0, 2, 0]).run()
        >>> events.sizes_.tolist(), events.durations_.tolist(), events.waiting_times_.tolist()
        ([4.0, 2.0], [2.0, 1.0], [2.0])
        >>> EventAnalysis('counts', window=1.0, end=3.0).prepare([0.1, 0.2, 1.5, 2.2]).run().counts_.tolist()
        [2.0, 1.0, 1.0]
        >>> laminar = EventAnalysis('laminar-durations', threshold=0.5)
        >>> laminar.prepare([1, 0, 0, 1, 0, 1]).run().durations_.tolist()
        [2.0, 1.0]
        >>> excursions = EventAnalysis('excursions').prepare([0, 1, 2, 1, 0, 3, 0]).run()
        >>> excursions.sizes_.tolist(), excursions.durations_.tolist()
        ([4.0, 3.0], [3.0, 1.0])
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> activity = np.zeros(20_000)
        >>> activity[0] = 100
        >>> for t in range(1, activity.size):
        ...     activity[t] = rng.poisson(0.9 * activity[t - 1] + 10)
        >>> round(EventAnalysis('branching').prepare(activity).run().branching_ratio_, 1)
        0.9
    """

    def __init__(
        self,
        kind: str = 'detect',
        variables: Sequence[str] | None = None,
        *,
        threshold: float = 0.0,
        window: float | None = None,
        start: float = 0.0,
        end: float | None = None,
        max_step: int = 40,
        dt: float = 1.0,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.threshold = threshold
        self.window = window
        self.start = start
        self.end = end
        self.max_step = max_step
        self.dt = dt

    def _check(self) -> None:
        """Validate the kind, the window of counts, the steps of branching, and that the kind reads every setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        if self.kind == 'counts' and (self.window is None or self.window <= 0):
            error_msg = f'{type(self).__name__} of kind counts has window={self.window!r}. Give a positive window.'
            raise AnalysisError(error_msg)
        if self.kind == 'branching' and (self.max_step < 1 or self.dt <= 0):
            error_msg = f'{type(self).__name__} has max_step={self.max_step!r} and dt={self.dt!r}. Use positive values.'
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the path kind for counts of event times, and the series kind otherwise."""
        return DataKind.PATH if self.kind == 'counts' else DataKind.SERIES

    def _prepare(self, source: Source) -> None:
        """Read the one series with its times, the times of the events, the records of excursions, or the trials."""
        records = self._records(source)
        if records:
            if self.kind != 'excursions':
                error_msg = f'{type(self).__name__} reads records of excursions for the kind excursions only.'
                raise AnalysisError(error_msg)
            self.times_: Array | None = None
            self.values_: Array | None = None
            self.labels_ = ['cascades']
            self.sizes_ = np.asarray([record.size for record in records], dtype=np.float64)
            self.durations_ = np.asarray([record.duration for record in records], dtype=np.float64)
            self.shapes_: list[Array] = []
            return
        name = self.variables[0] if self.variables else None
        if self.kind == 'counts':
            if isinstance(source, Trajectory):
                path = self._read_path(source, name)
                self.times_ = path.times[1:]
                self.labels_ = [path.label]
            else:
                self.times_ = np.asarray(source, dtype=float)
                self.labels_ = [name or 'events']
            self.values_ = self.times_[:, None]
            return
        series = self._read_series(source, self.variables)
        if series.values.shape[1] != 1 and self.kind != 'branching':
            error_msg = (
                f'{type(self).__name__} reads one series, but the source has {series.values.shape[1]}: '
                f'{", ".join(series.labels)}. Pick one with `variables`.'
            )
            raise AnalysisError(error_msg)
        self.times_ = np.asarray(series.times, dtype=float)
        self.values_ = series.values
        self.labels_ = series.labels

    def _run(self, inputs: object = None) -> None:
        """Compute the events, the counts, the laminar durations, the excursions, or the branching ratio."""
        if self.kind == 'excursions':
            if self.values_ is not None:
                self.shapes_ = self._excursions(np.asarray(self.values_[:, 0], dtype=np.float64), self.threshold)
                self.sizes_ = np.asarray([shape.sum() for shape in self.shapes_], dtype=np.float64)
                self.durations_ = np.asarray([shape.size for shape in self.shapes_], dtype=np.float64)
            return
        if self.values_ is None or self.times_ is None:
            error_msg = f'{type(self).__name__} reads records of excursions for the kind excursions only.'
            raise AnalysisError(error_msg)
        if self.kind == 'branching':
            self.branching_ratio_ = self._branching_ratio(
                np.asarray(self.values_, dtype=np.float64),
                self.max_step,
                self.dt,
            )
            return
        if self.kind == 'counts':
            self.counts_ = self._event_counts(self.times_, float(self.window or 0), self.start, self.end)
            return
        array = self._read_real(self.values_[:, 0], 'values')
        if self.kind == 'laminar-durations':
            self.durations_ = self._laminar_durations(array, self.times_, self.threshold)
            return
        self.starts_, self.sizes_, self.durations_, self.peaks_, self.waiting_times_ = self._detect_events(
            array,
            self.times_,
            self.threshold,
        )

    @staticmethod
    def _step(clock: Array) -> float:
        """Return the median time between samples, 1 for a single sample."""
        return float(np.median(np.diff(clock))) if clock.size > 1 else 1.0

    @staticmethod
    def _detect_events(array: Array, clock: Array, threshold: float) -> tuple[Array, Array, Array, Array, Array]:
        """Return the starts, sizes, durations, peaks, and waiting times of the runs of values above a threshold."""
        step = EventAnalysis._step(clock)
        active = np.concatenate([[False], array > threshold, [False]])
        edges = np.diff(active.astype(int))
        first, last = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        sizes = np.asarray([array[start:end].sum() for start, end in zip(first, last, strict=True)])
        peaks = np.asarray([array[start:end].max() for start, end in zip(first, last, strict=True)])
        durations = (clock[last - 1] - clock[first] + step) if first.size else np.zeros(0)
        waiting = clock[first[1:]] - clock[last[:-1] - 1] - step if first.size > 1 else np.zeros(0)
        return clock[first], sizes, np.asarray(durations, dtype=float), peaks, np.asarray(waiting, dtype=float)

    @staticmethod
    def _event_counts(times: Array, window: float, start: float, end: float | None) -> Array:
        """Return the number of events in consecutive windows of equal length."""
        last = float(times.max()) if end is None else end
        count = int((last - start) // window)
        edges = start + window * np.arange(count + 1)
        return np.histogram(times, bins=edges)[0].astype(float)

    @staticmethod
    def _laminar_durations(values: Array, clock: Array, threshold: float) -> Array:
        """Return the durations of the complete runs of values within a threshold of 0, between two bursts."""
        array = np.abs(values)
        step = EventAnalysis._step(clock)
        laminar = np.concatenate([[False], array <= threshold, [False]]).astype(int)
        starts, ends = np.flatnonzero(np.diff(laminar) == 1), np.flatnonzero(np.diff(laminar) == -1)
        complete = (starts > 0) & (ends < array.size)
        return clock[ends[complete] - 1] - clock[starts[complete]] + step

    @staticmethod
    def _records(source: object) -> list[CascadeRecordSpec]:
        """Return the records of excursions of a source, each with a size and a duration, or nothing for values."""
        if isinstance(source, np.ndarray | Trajectory) or not isinstance(source, Sequence):
            return []
        records = [item for item in source if isinstance(item, CascadeRecordSpec)]
        return records if records and len(records) == len(source) else []

    @staticmethod
    def _excursions(values: Array, threshold: float) -> list[Array]:
        """Return the profile of every maximal run of values above a threshold, its values minus the threshold."""
        excess = values - threshold
        active = np.concatenate([[False], excess > 0, [False]]).astype(int)
        starts, ends = np.flatnonzero(np.diff(active) == 1), np.flatnonzero(np.diff(active) == -1)
        return [excess[start:end] for start, end in zip(starts, ends, strict=True)]

    @staticmethod
    def _branching_ratio(values: Array, max_step: int, dt: float) -> float:
        """Return the branching ratio of trials, one per column, by the multistep regression of `mrestimator`."""
        mrestimator = BaseAnalysis._import('mrestimator')
        coefficients = mrestimator.coefficients(
            values.T,
            method='ts',
            steps=(1, max_step),
            dt=dt,
            dtunit='step',
            numboot=0,
        )
        result = mrestimator.fit(coefficients, fitfunc=mrestimator.f_exponential_offset, numboot=0)
        return float(result.mre)
