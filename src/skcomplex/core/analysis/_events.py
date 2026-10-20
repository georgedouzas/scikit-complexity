"""Detect events in series, such as avalanches and bursts, with their sizes, durations, and waiting times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._kinds import DataKind, analysis
from ._utils import real_array

type Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Events:
    """Events of a series: runs of consecutive values above a threshold.

    Args:
        starts:
            The time or index at which every event starts.

        sizes:
            The sum of the values of every event.

        durations:
            The time or the number of steps every event lasts.

        peaks:
            The largest value of every event.

        waiting_times:
            The time or the number of steps between the end of every event and the start of the next.
    """

    starts: Array
    sizes: Array
    durations: Array
    peaks: Array
    waiting_times: Array


@analysis(DataKind.SERIES)
def detect_events(values: npt.ArrayLike, threshold: float = 0.0, times: npt.ArrayLike | None = None) -> Events:
    """Return the events of a series, every maximal run of consecutive values above a threshold.

    For avalanches recorded as topplings per drive step, an event is an avalanche and its size is the total number of
    topplings.

    Args:
        values:
            The series.

        threshold:
            The value an event must exceed.

        times:
            The times of the values, or `None` to measure starts, durations, and waiting times in steps.

    Returns:
        events:
            The starts, sizes, durations, peaks, and waiting times.

    Examples:
        >>> from skcomplex.core.analysis import detect_events
        >>> events = detect_events([0, 3, 1, 0, 0, 2, 0])
        >>> events.sizes.tolist(), events.durations.tolist(), events.waiting_times.tolist()
        ([4.0, 2.0], [2.0, 1.0], [2.0])
    """
    array = real_array(values, 'values')
    clock = np.arange(array.size, dtype=float) if times is None else np.asarray(times, dtype=float)
    step = float(np.median(np.diff(clock))) if clock.size > 1 else 1.0
    active = np.concatenate([[False], array > threshold, [False]])
    edges = np.diff(active.astype(int))
    first, last = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    sizes = np.asarray([array[start:end].sum() for start, end in zip(first, last, strict=True)])
    peaks = np.asarray([array[start:end].max() for start, end in zip(first, last, strict=True)])
    durations = (clock[last - 1] - clock[first] + step) if first.size else np.zeros(0)
    waiting = clock[first[1:]] - clock[last[:-1] - 1] - step if first.size > 1 else np.zeros(0)
    return Events(clock[first], sizes, np.asarray(durations, dtype=float), peaks, np.asarray(waiting, dtype=float))
