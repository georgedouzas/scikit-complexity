"""Measure fluctuations: Fano factors, counts in windows, relative fluctuations, and Kramers-Moyal coefficients."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..base import AnalysisError
from ._kinds import DataKind, analysis
from ._lazy import module

type Array = npt.NDArray[np.float64]


@analysis(DataKind.SERIES)
def fano_factor(values: npt.ArrayLike) -> float:
    """Return the Fano factor, the variance over the mean, of counts or copy numbers.

    Args:
        values:
            The counts.

    Returns:
        factor:
            1 for Poisson counts, above 1 for bursty and below 1 for regular ones.

    Raises:
        AnalysisError:
            The mean is 0.

    Examples:
        >>> from skcomplex.core.analysis import fano_factor
        >>> fano_factor([1, 3, 1, 3])
        0.5
    """
    array = np.asarray(values, dtype=float)
    if array.mean() == 0:
        error_msg = 'The counts have mean 0, so their Fano factor is undefined.'
        raise AnalysisError(error_msg)
    return float(array.var() / array.mean())


@analysis(DataKind.PATH)
def event_counts(event_times: npt.ArrayLike, window: float, *, start: float = 0.0, end: float | None = None) -> Array:
    """Return the number of events in consecutive windows of equal length.

    Args:
        event_times:
            The times of the events.

        window:
            The length of every window.

        start:
            The start of the first window.

        end:
            The end of the last full window, or `None` for the last event.

    Returns:
        counts:
            The number of events in every window.

    Examples:
        >>> from skcomplex.core.analysis import event_counts
        >>> event_counts([0.1, 0.2, 1.5, 2.2], 1.0, end=3.0).tolist()
        [2.0, 1.0, 1.0]
    """
    times = np.asarray(event_times, dtype=float)
    last = float(times.max()) if end is None else end
    count = int((last - start) // window)
    edges = start + window * np.arange(count + 1)
    return np.histogram(times, bins=edges)[0].astype(float)


@dataclass(frozen=True)
class RelativeFluctuations:
    """Standard deviation over mean of a quantity against system size, and its scaling exponent.

    Args:
        sizes:
            The system sizes.

        ratios:
            The standard deviation over the mean at every size.

        exponent:
            The slope of `log ratio` against `log size`, -1/2 for independent parts.
    """

    sizes: Array
    ratios: Array
    exponent: float


@analysis(DataKind.ENSEMBLE)
def relative_fluctuations(samples: Mapping[float, npt.ArrayLike]) -> RelativeFluctuations:
    """Return the relative fluctuations of a quantity sampled at several system sizes and their scaling exponent.

    Args:
        samples:
            The samples of the quantity at every size, such as particle numbers in subvolumes.

    Returns:
        fluctuations:
            The sizes, the ratios, and the exponent.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import relative_fluctuations
        >>> rng = np.random.default_rng(0)
        >>> samples = {size: rng.poisson(size, 5000) for size in (10, 100, 1000)}
        >>> round(relative_fluctuations(samples).exponent, 1)
        -0.5
    """
    sizes = np.asarray(sorted(samples), dtype=np.float64)
    arrays = [np.asarray(samples[size], dtype=np.float64) for size in sorted(samples)]
    ratios = np.asarray([array.std() / array.mean() for array in arrays], dtype=np.float64)
    exponent = float(np.polyfit(np.log(sizes), np.log(ratios), 1)[0])
    return RelativeFluctuations(sizes, ratios, exponent)


@dataclass(frozen=True)
class KramersMoyal:
    """Drift and diffusion coefficients of a series as functions of its value.

    Args:
        positions:
            The centres of the bins of values.

        drift:
            The first coefficient `<dx | x> / dt` in every bin.

        diffusion:
            The second coefficient `<dx² | x> / (2 dt)` in every bin, so `dx = a dt + sqrt(2 D) dW` gives `D`.
    """

    positions: Array
    drift: Array
    diffusion: Array


@analysis(DataKind.SERIES)
def kramers_moyal(
    values: npt.ArrayLike,
    dt: float = 1.0,
    *,
    bins: int = 50,
    bandwidth: float | None = None,
) -> KramersMoyal:
    """Return the Kramers-Moyal drift and diffusion of a series, estimated with kernels from its increments.

    Args:
        values:
            The series.

        dt:
            The time between samples.

        bins:
            The number of bins of values.

        bandwidth:
            The width of the Epanechnikov kernel, or `None` for Silverman's rule.

    Returns:
        coefficients:
            The bin centres, the drift, and the diffusion.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import kramers_moyal
        >>> rng = np.random.default_rng(0)
        >>> x = np.zeros(100_000)
        >>> for t in range(1, x.size):
        ...     x[t] = x[t - 1] - x[t - 1] * 0.01 + np.sqrt(0.02) * rng.standard_normal()
        >>> coefficients = kramers_moyal(x, 0.01, bins=21)
        >>> round(float(coefficients.diffusion[10]), 1)
        1.0
    """
    array = np.asarray(values, dtype=float)
    moments, edges = module('kramersmoyal').km(
        array,
        bins=np.asarray([bins]),
        powers=np.asarray([0, 1, 2]),
        bw=bandwidth,
    )
    centres = np.asarray(edges[0], dtype=float)
    return KramersMoyal(centres, moments[1] / dt, moments[2] / dt)
