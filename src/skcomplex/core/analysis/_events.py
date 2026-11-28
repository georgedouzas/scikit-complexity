"""Detect and measure events in series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import contextlib
import io
import logging
import logging.handlers
import queue
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..engine.problems._cascade_kernel import _CascadeRecordSpec
from ..engine.runtime import Trajectory
from ..enums import DataKind
from ..errors import AnalysisError
from ..types import Array
from ._analysis_tags import AnalysisTags
from ._base import _check_choice, _check_settings, _read_path, _read_real, _read_series
from ._utils import load_module
from .types import DataSource

SETTINGS = {
    'detect': {'threshold'},
    'counts': {'window', 'start', 'end'},
    'laminar-durations': {'threshold'},
    'excursions': {'threshold'},
    'branching': {'max_step', 'dt'},
    'activity': set(),
}
KINDS = ('detect', 'counts', 'laminar-durations', 'excursions', 'branching', 'activity')
RECORD_KINDS = ('excursions', 'activity')
TAGS = AnalysisTags(DataKind.SERIES)
KIND_TAGS = {
    'counts': AnalysisTags(DataKind.PATH),
}


@dataclass(frozen=True)
class EventsResult:
    """A result of `events`, the events of one series or of event times.

    It holds the series it read and what the kind measures: the start, size, duration, and peak of every event, the
    counts in windows, the laminar phases, the excursions, the activity of cascades in fast time, or the branching
    ratio, such as the sizes 4 and 2 of the
    two bursts of `[0, 3, 1, 0, 0, 2, 0]`. It reports the crashes of a market, the spikes of a neuron, and the floods
    of a river.

    Read more in the user guide, under Analyse.

    Args:
        times:
            The time of every value, of every event for `'counts'`, or the index of every round for `'activity'`.

        values:
            The series, as one column, one column per trial for `'branching'`, the activity of every round for
            `'activity'`, or `None` for excursions of records.

        labels:
            The label of the series or the state.

        starts:
            The time at which every event starts.

        sizes:
            The sum of the values of every event.

        durations:
            The time every event lasts for `'detect'`, every laminar phase lasts for `'laminar-durations'`, or the
            number of samples of every excursion for `'excursions'`.

        shapes:
            The profile of every excursion, of a series or of records.

        peaks:
            The largest value of every event.

        waiting_times:
            The time between the end of every event and the start of the next.

        counts:
            The number of events in every window.

        branching_ratio:
            The branching ratio `m`, 1 at criticality.

    See Also:
        events: Returns this result.

    Examples:
        >>> from skcomplex.core.analysis import events
        >>> events([0, 3, 1, 0, 0, 2, 0]).sizes.tolist()
        [4.0, 2.0]
    """

    times: Array | None
    values: Array | None
    labels: list[str]
    starts: Array | None = None
    sizes: Array | None = None
    durations: Array | None = None
    shapes: list[Array] | None = None
    peaks: Array | None = None
    waiting_times: Array | None = None
    counts: Array | None = None
    branching_ratio: float | None = None


def events(
    source: DataSource,
    *,
    kind: str = 'detect',
    variables: Sequence[str] | None = None,
    threshold: float = 0.0,
    window: float | None = None,
    start: float = 0.0,
    end: float | None = None,
    max_step: int = 40,
    dt: float = 1.0,
) -> EventsResult:
    """Return the events of one series or of event times, of one kind.

    An event is a run of consecutive values above a threshold, such as the bursts of sizes 4 and 2 in
    `[0, 3, 1, 0, 0, 2, 0]`, with the waiting time between them. Times are the times of a trajectory, and the step
    indices of an array. It finds the crashes of a market, the spikes of a neuron, the avalanches of a sandpile, and
    the floods of a river.

    Read more in the user guide, under Analyse.

    Args:
        source:
            The trajectory, an array that holds one series, or one column per trial for `'branching'`, the times of the
            events for `'counts'`, or the records of cascades of a trajectory, its `events`, for `'excursions'` and
            `'activity'`.

        kind:
            What to return. `'detect'` gives every maximal run of values above `threshold`, with its start, size,
            duration, and peak, and the waiting times between runs. `'counts'` gives the number of events in
            consecutive windows of length `window`, from the times of the events or the jump times of a discrete
            state of a trajectory. `'laminar-durations'` gives the durations of the complete runs of values within
            `threshold` of 0 between two bursts, the quiet phases of on and off intermittency. `'excursions'` gives the
            profile of every run above `threshold`, its values minus the threshold, with its size, the sum of the
            profile, and its duration, the number of its samples. The cascade records of a lattice, such as the
            avalanches of a sandpile, are excursions already, whose profiles are the active sites of every round.
            `'activity'` gives the activity of every round of every cascade of the records in fast time, in the order of
            the records, each cascade after one round of activity 0 for the drive that starts it, such as
            `0, 1, 2, 0, 3` for the profiles `(1, 2)` and `(3,)`. A step whose drive starts no cascade leaves no record,
            so it adds no round. `'branching'` gives the branching ratio by the
            multistep regression of `mrestimator`, which fits `r_k ∝ m^k` to the slopes `r_k` of the values `k` steps
            apart and stays unbiased when only some of the units are observed, with every series one trial. It prints
            nothing, and raises the messages `mrestimator` logs, such as an autocorrelation time long against the
            trial, as a `RuntimeWarning`.

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

    Returns:
        result:
            The values read and the events, counts, durations, excursions, or branching ratio of the kind.

    Raises:
        AnalysisError:
            The kind is unknown, a setting the kind does not read is changed, a setting is out of range, or the source
            does not suit the kind.

    See Also:
        distribution: Fits the distribution of the sizes or the waiting times.
        fit_law: Fits a power law to the sizes of avalanches.

    Examples:
        >>> from skcomplex.core.analysis import events
        >>> found = events([0, 3, 1, 0, 0, 2, 0])
        >>> found.sizes.tolist(), found.durations.tolist(), found.waiting_times.tolist()
        ([4.0, 2.0], [2.0, 1.0], [2.0])
    """
    _check_choice('events', 'kind', kind, KINDS)
    _check_settings(
        events,
        'kind',
        SETTINGS[kind] | {'variables'},
        kind=kind,
        variables=variables,
        threshold=threshold,
        window=window,
        start=start,
        end=end,
        max_step=max_step,
        dt=dt,
    )
    _check(kind, window, max_step, dt)
    records = _records(source)
    if records:
        if kind not in RECORD_KINDS:
            error_msg = 'events reads records of cascades for the kinds excursions and activity only.'
            raise AnalysisError(error_msg)
        if kind == 'activity':
            series = _fast_time_activity(records)
            return EventsResult(np.arange(series.size, dtype=np.float64), series[:, None], ['activity'])
        return EventsResult(
            None,
            None,
            ['cascades'],
            sizes=np.asarray([record.size for record in records], dtype=np.float64),
            durations=np.asarray([record.duration for record in records], dtype=np.float64),
            shapes=[np.asarray(record.profile, dtype=np.float64) for record in records],
        )
    if kind == 'activity':
        error_msg = 'events of kind activity reads the records of cascades of a run, such as trajectory.events.'
        raise AnalysisError(error_msg)
    name = variables[0] if variables else None
    if kind == 'counts':
        if isinstance(source, Trajectory):
            path = _read_path(source, name)
            times, labels = path.times[1:], [path.label]
        else:
            times, labels = np.asarray(source, dtype=float), [name or 'events']
        counts = _event_counts(times, float(window or 0), start, end)
        return EventsResult(times, times[:, None], labels, counts=counts)
    series = _read_series(source, variables)
    if series.values.shape[1] != 1 and kind != 'branching':
        error_msg = (
            f'events reads one series, but the source has {series.values.shape[1]}: {", ".join(series.labels)}. Pick '
            'one with `variables`.'
        )
        raise AnalysisError(error_msg)
    times, values, labels = np.asarray(series.times, dtype=float), series.values, series.labels
    if kind == 'excursions':
        shapes = _excursions(np.asarray(values[:, 0], dtype=np.float64), threshold)
        return EventsResult(
            times,
            values,
            labels,
            sizes=np.asarray([shape.sum() for shape in shapes], dtype=np.float64),
            durations=np.asarray([shape.size for shape in shapes], dtype=np.float64),
            shapes=shapes,
        )
    if kind == 'branching':
        ratio = _branching_ratio(np.asarray(values, dtype=np.float64), max_step, dt)
        return EventsResult(times, values, labels, branching_ratio=ratio)
    array = _read_real(values[:, 0], 'values')
    if kind == 'laminar-durations':
        return EventsResult(times, values, labels, durations=_laminar_durations(array, times, threshold))
    starts, sizes, durations, peaks, waiting_times = _detect_events(array, times, threshold)
    return EventsResult(
        times,
        values,
        labels,
        starts=starts,
        sizes=sizes,
        durations=durations,
        peaks=peaks,
        waiting_times=waiting_times,
    )


def _step(clock: Array) -> float:
    """Return the median time between samples, 1 for a single sample."""
    return float(np.median(np.diff(clock))) if clock.size > 1 else 1.0


def _detect_events(array: Array, clock: Array, threshold: float) -> tuple[Array, Array, Array, Array, Array]:
    """Return the starts, sizes, durations, peaks, and waiting times of events.

    The events are the runs of values above a threshold.
    """
    step = _step(clock)
    active = np.concatenate([[False], array > threshold, [False]])
    edges = np.diff(active.astype(int))
    first, last = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    sizes = np.asarray([array[start:end].sum() for start, end in zip(first, last, strict=True)])
    peaks = np.asarray([array[start:end].max() for start, end in zip(first, last, strict=True)])
    durations = (clock[last - 1] - clock[first] + step) if first.size else np.zeros(0)
    waiting = clock[first[1:]] - clock[last[:-1] - 1] - step if first.size > 1 else np.zeros(0)
    return clock[first], sizes, np.asarray(durations, dtype=float), peaks, np.asarray(waiting, dtype=float)


def _event_counts(times: Array, window: float, start: float, end: float | None) -> Array:
    """Return the number of events in consecutive windows of equal length."""
    last = float(times.max()) if end is None else end
    count = int((last - start) // window)
    edges = start + window * np.arange(count + 1)
    return np.histogram(times, bins=edges)[0].astype(float)


def _laminar_durations(values: Array, clock: Array, threshold: float) -> Array:
    """Return the durations of the complete runs of values within a threshold of 0.

    The runs lie between two bursts.
    """
    array = np.abs(values)
    step = _step(clock)
    laminar = np.concatenate([[False], array <= threshold, [False]]).astype(int)
    starts, ends = np.flatnonzero(np.diff(laminar) == 1), np.flatnonzero(np.diff(laminar) == -1)
    complete = (starts > 0) & (ends < array.size)
    return clock[ends[complete] - 1] - clock[starts[complete]] + step


def _records(source: object) -> list[_CascadeRecordSpec]:
    """Return the records of excursions of a source.

    Each record has a size and a duration, and values give nothing.
    """
    if isinstance(source, np.ndarray | Trajectory) or not isinstance(source, Sequence):
        return []
    records = [item for item in source if isinstance(item, _CascadeRecordSpec)]
    return records if records and len(records) == len(source) else []


def _fast_time_activity(records: Sequence[_CascadeRecordSpec]) -> Array:
    """Return the activity of every round of the cascades of records, each after a quiet round of its drive."""
    rounds = [value for record in records for value in (0, *record.profile)]
    return np.asarray(rounds, dtype=np.float64)


def _excursions(values: Array, threshold: float) -> list[Array]:
    """Return the profile of every maximal run of values above a threshold.

    The profile is its values minus the threshold.
    """
    excess = values - threshold
    active = np.concatenate([[False], excess > 0, [False]]).astype(int)
    starts, ends = np.flatnonzero(np.diff(active) == 1), np.flatnonzero(np.diff(active) == -1)
    return [excess[start:end] for start, end in zip(starts, ends, strict=True)]


def _estimate_branching_ratio(values: Array, max_step: int, dt: float) -> float:
    """Return the branching ratio of trials by the regression of `mrestimator`."""
    mrestimator = load_module('mrestimator')
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


def _branching_ratio(values: Array, max_step: int, dt: float) -> float:
    """Return the branching ratio of trials, one per column.

    The progress bars of `mrestimator` are silenced, and the warnings it logs are raised as Python warnings.
    """
    logger = logging.getLogger('mrestimator')
    records: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
    handler = logging.handlers.QueueHandler(records)
    handler.setLevel(logging.WARNING)
    propagate = logger.propagate
    logger.addHandler(handler)
    logger.propagate = False
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ratio = _estimate_branching_ratio(values, max_step, dt)
    finally:
        logger.removeHandler(handler)
        logger.propagate = propagate
    while not records.empty():
        warnings.warn(records.get().getMessage(), RuntimeWarning, stacklevel=3)
    return ratio


def _check(kind: str, window: float | None, max_step: int, dt: float) -> None:
    """Raise an error for a window of counts or steps of branching out of range."""
    if kind == 'counts' and (window is None or window <= 0):
        error_msg = f'events of kind counts has window={window!r}. Give a positive window.'
        raise AnalysisError(error_msg)
    if kind == 'branching' and (max_step < 1 or dt <= 0):
        error_msg = f'events has max_step={max_step!r} and dt={dt!r}. Use positive values.'
        raise AnalysisError(error_msg)
