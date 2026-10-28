"""Detect events in series, such as avalanches and bursts, count events in windows, and time laminar phases."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt

from ..core.errors import AnalysisError
from ..engine.runtime import Trajectory
from ._base import BaseAnalysis, Source
from ._kinds import DataKind

type Array = npt.NDArray[np.float64]
KINDS = ('detect', 'counts', 'laminar-durations')
SETTINGS = {
    'detect': {'threshold'},
    'counts': {'window', 'start', 'end'},
    'laminar-durations': {'threshold'},
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

    Attributes:
        times_:
            The time of every value, or of every event for `'counts'`.

        values_:
            The series, as one column.

        labels_:
            The label of the series or the state.

        starts_:
            The time at which every event starts.

        sizes_:
            The sum of the values of every event.

        durations_:
            The time every event lasts for `'detect'`, or every laminar phase lasts for `'laminar-durations'`.

        peaks_:
            The largest value of every event.

        waiting_times_:
            The time between the end of every event and the start of the next.

        counts_:
            The number of events in every window.

    Examples:
        >>> from skcomplex.analysis import EventAnalysis
        >>> events = EventAnalysis().check().prepare([0, 3, 1, 0, 0, 2, 0]).run()
        >>> events.sizes_.tolist(), events.durations_.tolist(), events.waiting_times_.tolist()
        ([4.0, 2.0], [2.0, 1.0], [2.0])
        >>> EventAnalysis('counts', window=1.0, end=3.0).check().prepare([0.1, 0.2, 1.5, 2.2]).run().counts_.tolist()
        [2.0, 1.0, 1.0]
        >>> laminar = EventAnalysis('laminar-durations', threshold=0.5)
        >>> laminar.check().prepare([1, 0, 0, 1, 0, 1]).run().durations_.tolist()
        [2.0, 1.0]
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
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.threshold = threshold
        self.window = window
        self.start = start
        self.end = end

    def _check(self) -> Self:
        """Validate the kind, the window of counts, and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        if self.kind == 'counts' and (self.window is None or self.window <= 0):
            error_msg = f'{type(self).__name__} of kind counts has window={self.window!r}. Give a positive window.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the path kind for counts of event times, and the series kind otherwise."""
        return DataKind.PATH if self.kind == 'counts' else DataKind.SERIES

    def _prepare(self, source: Source) -> Self:
        """Read the one series with its times, or the times of the events."""
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
            return self
        series = self._read_series(source, self.variables)
        if series.values.shape[1] != 1:
            error_msg = (
                f'{type(self).__name__} reads one series, but the source has {series.values.shape[1]}: '
                f'{", ".join(series.labels)}. Pick one with `variables`.'
            )
            raise AnalysisError(error_msg)
        self.times_ = np.asarray(series.times, dtype=float)
        self.values_ = series.values
        self.labels_ = series.labels
        return self

    def _run(self) -> Self:
        """Compute the events, the counts, or the laminar durations."""
        if self.kind == 'counts':
            self.counts_ = self._event_counts(self.times_, float(self.window or 0), self.start, self.end)
            return self
        array = self._read_real(self.values_[:, 0], 'values')
        if self.kind == 'laminar-durations':
            self.durations_ = self._laminar_durations(array, self.times_, self.threshold)
            return self
        self.starts_, self.sizes_, self.durations_, self.peaks_, self.waiting_times_ = self._detect_events(
            array,
            self.times_,
            self.threshold,
        )
        return self

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
