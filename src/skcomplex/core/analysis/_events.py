"""Detect and measure events in series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..specs import AnalysisTags, CascadeRecordSpec
from ..types import Array, DataKind
from ._base import _check_choice, _check_settings, _read_path, _read_real, _read_series
from ._utils import load_module
from .types import Source

SETTINGS = {
    'detect': {'threshold'},
    'counts': {'window', 'start', 'end'},
    'laminar-durations': {'threshold'},
    'excursions': {'threshold'},
    'branching': {'max_step', 'dt'},
}
KINDS = ('detect', 'counts', 'laminar-durations', 'excursions', 'branching')
TAGS = AnalysisTags(DataKind.SERIES)
KIND_TAGS = {
    'counts': AnalysisTags(DataKind.PATH),
}


@dataclass(frozen=True)
class EventsResult:
    """A result of `events`, the events of one series or of event times.

    Args:
        times:
            The time of every value, or of every event for `'counts'`.

        values:
            The series, as one column, or one column per trial for `'branching'`, or `None` for records.

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
            The profile of every excursion of a series, empty for records.

        peaks:
            The largest value of every event.

        waiting_times:
            The time between the end of every event and the start of the next.

        counts:
            The number of events in every window.

        branching_ratio:
            The branching ratio `m`, 1 at criticality.
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
    source: Source,
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

    It appears in many fields. It finds the crashes of a market, the spikes of a neuron, and the floods of a river.

    Args:
        source:
            The trajectory, an array that holds one series, or one column per trial for `'branching'`, the times of the
            events for `'counts'`, or records of excursions for `'excursions'`.

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

    Returns:
        result:
            The values read and the events, counts, durations, excursions, or branching ratio of the kind.

    Raises:
        AnalysisError:
            The kind is unknown, a setting the kind does not read is changed, a setting is out of range, or the source
            does not suit the kind.

    Examples:
        >>> from skcomplex.core.analysis import events
        >>> found = events([0, 3, 1, 0, 0, 2, 0])
        >>> found.sizes.tolist(), found.durations.tolist(), found.waiting_times.tolist()
        ([4.0, 2.0], [2.0, 1.0], [2.0])
        >>> events([0.1, 0.2, 1.5, 2.2], kind='counts', window=1.0, end=3.0).counts.tolist()
        [2.0, 1.0, 1.0]
        >>> events([1, 0, 0, 1, 0, 1], kind='laminar-durations', threshold=0.5).durations.tolist()
        [2.0, 1.0]
        >>> excursions = events([0, 1, 2, 1, 0, 3, 0], kind='excursions')
        >>> excursions.sizes.tolist(), excursions.durations.tolist()
        ([4.0, 3.0], [3.0, 1.0])
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> activity = np.zeros(20_000)
        >>> activity[0] = 100
        >>> for t in range(1, activity.size):
        ...     activity[t] = rng.poisson(0.9 * activity[t - 1] + 10)
        >>> round(events(activity, kind='branching').branching_ratio, 1)
        0.9
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
        if kind != 'excursions':
            error_msg = 'events reads records of excursions for the kind excursions only.'
            raise AnalysisError(error_msg)
        return EventsResult(
            None,
            None,
            ['cascades'],
            sizes=np.asarray([record.size for record in records], dtype=np.float64),
            durations=np.asarray([record.duration for record in records], dtype=np.float64),
            shapes=[],
        )
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


def _records(source: object) -> list[CascadeRecordSpec]:
    """Return the records of excursions of a source.

    Each record has a size and a duration, and values give nothing.
    """
    if isinstance(source, np.ndarray | Trajectory) or not isinstance(source, Sequence):
        return []
    records = [item for item in source if isinstance(item, CascadeRecordSpec)]
    return records if records and len(records) == len(source) else []


def _excursions(values: Array, threshold: float) -> list[Array]:
    """Return the profile of every maximal run of values above a threshold.

    The profile is its values minus the threshold.
    """
    excess = values - threshold
    active = np.concatenate([[False], excess > 0, [False]]).astype(int)
    starts, ends = np.flatnonzero(np.diff(active) == 1), np.flatnonzero(np.diff(active) == -1)
    return [excess[start:end] for start, end in zip(starts, ends, strict=True)]


def _branching_ratio(values: Array, max_step: int, dt: float) -> float:
    """Return the branching ratio of trials, one per column.

    It is found by the multistep regression of `mrestimator`.
    """
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


def _check(kind: str, window: float | None, max_step: int, dt: float) -> None:
    """Raise an error for a window of counts or steps of branching out of range."""
    if kind == 'counts' and (window is None or window <= 0):
        error_msg = f'events of kind counts has window={window!r}. Give a positive window.'
        raise AnalysisError(error_msg)
    if kind == 'branching' and (max_step < 1 or dt <= 0):
        error_msg = f'events has max_step={max_step!r} and dt={dt!r}. Use positive values.'
        raise AnalysisError(error_msg)
