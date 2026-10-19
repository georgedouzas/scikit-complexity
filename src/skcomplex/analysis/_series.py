"""Measure memory, spectra, diffusion, and early warnings of tipping in time series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import signal, stats

from ..core import AnalysisError

type Array = npt.NDArray[np.float64]


def autocorrelation(values: npt.ArrayLike, max_lag: int | None = None) -> Array:
    """Return the autocorrelation of a series at lags 0 to `max_lag`, computed with the FFT.

    Args:
        values:
            The series.

        max_lag:
            The largest lag, or `None` for half the series length.

    Returns:
        autocorrelation:
            The autocorrelation per lag, 1 at lag 0.

    Examples:
        >>> from skcomplex.analysis import autocorrelation
        >>> autocorrelation([1.0, -1.0, 1.0, -1.0], max_lag=1).round(2).tolist()
        [1.0, -0.75]
    """
    array = np.asarray(values, dtype=float)
    array = array - array.mean()
    size = array.size
    spectrum = np.fft.rfft(array, n=2 * size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum))[:size]
    correlation /= correlation[0] if correlation[0] else 1.0
    return correlation[: (size // 2 if max_lag is None else max_lag) + 1]


def power_spectrum(values: npt.ArrayLike, dt: float = 1.0) -> tuple[Array, Array]:
    """Return the frequencies and the power spectral density of a series by Welch's method.

    Args:
        values:
            The series.

        dt:
            The time between samples.

    Returns:
        spectrum:
            The frequencies and the density at each of them.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import power_spectrum
        >>> times = np.arange(4096) * 0.01
        >>> frequencies, density = power_spectrum(np.sin(2 * np.pi * 5 * times), dt=0.01)
        >>> round(float(frequencies[density.argmax()]))
        5
    """
    array = np.asarray(values, dtype=float)
    return signal.welch(array, fs=1 / dt, nperseg=min(array.size, 1024))


@dataclass(frozen=True)
class Displacement:
    """Mean squared displacement over lags and its scaling exponent.

    Args:
        lags:
            The lags in time.

        msd:
            The mean squared displacement at every lag.

        exponent:
            The slope of `log msd` against `log lag`, 1 for normal diffusion, below 1 for subdiffusion, and above 1
            for superdiffusion.
    """

    lags: Array
    msd: Array
    exponent: float


def mean_squared_displacement(
    positions: npt.ArrayLike,
    dt: float = 1.0,
    lags: npt.ArrayLike | None = None,
) -> Displacement:
    """Return the mean squared displacement of particles over lags and its scaling exponent.

    Args:
        positions:
            The positions with time as the first axis, then particles, then optionally dimensions.

        dt:
            The time between frames.

        lags:
            The lags in frames, or `None` for 20 lags spaced logarithmically up to a quarter of the frames.

    Returns:
        displacement:
            The lags, the mean squared displacement, and its exponent.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import mean_squared_displacement
        >>> walks = np.cumsum(np.random.default_rng(0).standard_normal((2000, 500)), axis=0)
        >>> round(mean_squared_displacement(walks).exponent, 1)
        1.0
    """
    array = np.asarray(positions, dtype=float)
    array = array.reshape(array.shape[0], array.shape[1] if array.ndim > 1 else 1, -1)
    frames = array.shape[0]
    steps = (
        np.unique(np.logspace(0, np.log10(max(frames // 4, 2)), 20).astype(int))
        if lags is None
        else np.asarray(lags, dtype=int)
    )
    msd = np.asarray([np.mean(np.sum((array[lag:] - array[:-lag]) ** 2, axis=-1)) for lag in steps])
    exponent = float(np.polyfit(np.log(steps * dt), np.log(msd), 1)[0])
    return Displacement(steps * dt, msd, exponent)


def hurst_exponent(values: npt.ArrayLike, scales: npt.ArrayLike | None = None, order: int = 1) -> float:
    """Return the Hurst exponent of a series by detrended fluctuation analysis.

    The series is integrated, cut into windows of every scale, detrended with a polynomial in every window, and the
    exponent is the slope of the log fluctuation against the log scale: 0.5 for uncorrelated noise, above 0.5 for
    persistent and below 0.5 for anti-persistent series.

    Args:
        values:
            The series.

        scales:
            The window sizes, or `None` for 20 sizes spaced logarithmically from 10 to a tenth of the series.

        order:
            The order of the detrending polynomial.

    Returns:
        hurst:
            The Hurst exponent.

    Raises:
        AnalysisError:
            The series is shorter than 100 values.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import hurst_exponent
        >>> round(hurst_exponent(np.random.default_rng(0).standard_normal(20000)), 1)
        0.5
    """
    array = np.asarray(values, dtype=float)
    if array.size < 100:  # noqa: PLR2004
        error_msg = f'Detrended fluctuation analysis needs at least 100 values, not {array.size}.'
        raise AnalysisError(error_msg)
    profile = np.cumsum(array - array.mean())
    sizes = (
        np.unique(np.logspace(1, np.log10(array.size // 10), 20).astype(int))
        if scales is None
        else np.asarray(scales, dtype=int)
    )
    fluctuations = []
    for size in sizes:
        count = profile.size // size
        windows = profile[: count * size].reshape(count, size)
        axis = np.arange(size)
        coefficients = np.polynomial.polynomial.polyfit(axis, windows.T, order)
        trend = np.polynomial.polynomial.polyval(axis, coefficients)
        fluctuations.append(np.sqrt(np.mean((windows - trend) ** 2)))
    return float(np.polyfit(np.log(sizes), np.log(fluctuations), 1)[0])


@dataclass(frozen=True)
class EarlyWarning:
    """Rolling indicators of critical slowing down before a tipping point.

    Args:
        variance:
            The variance in every window, `nan` before the first full window.

        autocorrelation:
            The lag-1 autocorrelation in every window, `nan` before the first full window.

        variance_trend:
            Kendall's tau of the variance against time, positive when it rises.

        autocorrelation_trend:
            Kendall's tau of the autocorrelation against time, positive when it rises.
    """

    variance: Array
    autocorrelation: Array
    variance_trend: float
    autocorrelation_trend: float


def early_warning(values: npt.ArrayLike, window: int, *, detrend: bool = True) -> EarlyWarning:
    """Return rolling variance and lag-1 autocorrelation, which rise as a system slows down before a tipping point.

    Args:
        values:
            The series.

        window:
            The number of values in every rolling window.

        detrend:
            Whether to remove a linear trend in every window first.

    Returns:
        warning:
            The rolling indicators and their trends.

    Raises:
        AnalysisError:
            The window is shorter than 3 values or longer than the series.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import early_warning
        >>> rng = np.random.default_rng(0)
        >>> x = np.zeros(2000)
        >>> for t in range(1, 2000):
        ...     x[t] = (0.2 + 0.79 * t / 2000) * x[t - 1] + rng.standard_normal()
        >>> early_warning(x, 200).autocorrelation_trend > 0.5
        True
    """
    array = np.asarray(values, dtype=float)
    if not 3 <= window <= array.size:  # noqa: PLR2004
        error_msg = f'window = {window} must lie between 3 and the length of the series, {array.size}.'
        raise AnalysisError(error_msg)
    variance = np.full(array.size, np.nan)
    correlation = np.full(array.size, np.nan)
    axis = np.arange(window)
    for end in range(window, array.size + 1):
        segment = array[end - window : end]
        if detrend:
            segment = segment - np.polyval(np.polyfit(axis, segment, 1), axis)
        variance[end - 1] = segment.var()
        correlation[end - 1] = np.corrcoef(segment[:-1], segment[1:])[0, 1]
    valid = ~np.isnan(variance)
    times = np.arange(array.size)[valid]
    return EarlyWarning(
        variance,
        correlation,
        float(stats.kendalltau(times, variance[valid]).statistic),
        float(stats.kendalltau(times, correlation[valid]).statistic),
    )
