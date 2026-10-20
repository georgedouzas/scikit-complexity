"""Measure memory, spectra, diffusion, and early warnings of tipping in time series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import nolds
import numpy as np
import numpy.typing as npt
import tidynamics
from scipy import signal, stats

from ..base import AnalysisError
from ._kinds import DataKind, analysis
from ._utils import real_array

type Array = npt.NDArray[np.float64]
MIN_FLUCTUATION_VALUES = 100
MIN_WINDOW = 3
HURST_METHODS = ('dfa', 'rs')
FIT_POINTS = 20


@analysis(DataKind.SERIES)
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
        >>> from skcomplex.core.analysis import autocorrelation
        >>> autocorrelation([1.0, -1.0, 1.0, -1.0], max_lag=1).round(2).tolist()
        [1.0, -0.75]
    """
    array = real_array(values, 'values')
    array = array - array.mean()
    size = array.size
    spectrum = np.fft.rfft(array, n=2 * size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum))[:size]
    correlation /= correlation[0] if correlation[0] else 1.0
    return correlation[: (size // 2 if max_lag is None else max_lag) + 1]


@analysis(DataKind.SERIES)
def autocorrelation_time(values: npt.ArrayLike, dt: float = 1.0) -> float:
    """Return the integrated autocorrelation time, `dt (1/2 + Σ r(k))` summed up to the first negative correlation.

    Args:
        values:
            The series.

        dt:
            The time between samples.

    Returns:
        time:
            The integrated autocorrelation time, `dt / 2` for white noise.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import autocorrelation_time
        >>> rng = np.random.default_rng(0)
        >>> x = np.zeros(50_000)
        >>> for t in range(1, x.size):
        ...     x[t] = 0.9 * x[t - 1] + rng.standard_normal()
        >>> 9 < autocorrelation_time(x) < 12
        True
    """
    correlation = autocorrelation(values)
    negative = np.flatnonzero(correlation <= 0)
    end = int(negative[0]) if negative.size else correlation.size
    return float(dt * (correlation[:end].sum() - 0.5))


@analysis(DataKind.SERIES)
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
        >>> from skcomplex.core.analysis import power_spectrum
        >>> times = np.arange(4096) * 0.01
        >>> frequencies, density = power_spectrum(np.sin(2 * np.pi * 5 * times), dt=0.01)
        >>> round(float(frequencies[density.argmax()]))
        5
    """
    array = real_array(values, 'values')
    return signal.welch(array, fs=1 / dt, nperseg=min(array.size, 1024))


def _particle_axes(positions: npt.ArrayLike) -> Array:
    """Return positions as frames by particles by dimensions."""
    array = real_array(positions, 'positions')
    return array.reshape(array.shape[0], array.shape[1] if array.ndim > 1 else 1, -1)


def _fit_lags(lags: Array, fit_range: tuple[float, float] | None) -> npt.NDArray[np.intp]:
    """Return the indices of about 20 lags spaced logarithmically within the fit range."""
    low, high = (lags[0], lags[-1] / 4) if fit_range is None else fit_range
    inside = np.flatnonzero((lags >= low) & (lags <= high))
    if inside.size < 2:  # noqa: PLR2004
        error_msg = f'The fit range {fit_range} holds fewer than 2 lags. Widen it within {lags[0]} to {lags[-1]}.'
        raise AnalysisError(error_msg)
    chosen = np.unique(np.geomspace(1, inside.size, FIT_POINTS).astype(int) - 1)
    return inside[chosen]


@dataclass(frozen=True)
class Displacement:
    """Mean squared displacement over lags, its scaling exponent, and the diffusion coefficient.

    Args:
        lags:
            The lags in time.

        msd:
            The mean squared displacement at every lag.

        exponent:
            The slope of `log msd` against `log lag` over the fit range, 1 for normal diffusion, below 1 for
            subdiffusion, and above 1 for superdiffusion.

        diffusion:
            The coefficient `D` of `msd = 2 d D lag` fitted over the fit range in `d` dimensions.
    """

    lags: Array
    msd: Array
    exponent: float
    diffusion: float = float('nan')


@analysis(DataKind.AGENTS)
def mean_squared_displacement(
    positions: npt.ArrayLike,
    dt: float = 1.0,
    lags: npt.ArrayLike | None = None,
    *,
    fit_range: tuple[float, float] | None = None,
) -> Displacement:
    """Return the mean squared displacement of particles over all lags, its scaling exponent, and the diffusion.

    The displacement of every particle is computed for all lags at once with the FFT, so positions must be unwrapped
    across periodic boundaries, as `snapshots_of` returns them.

    Args:
        positions:
            The unwrapped positions with time as the first axis, then particles, then optionally dimensions.

        dt:
            The time between frames.

        lags:
            The lags in frames to return, or `None` for every lag from 1.

        fit_range:
            The smallest and largest lags in time of the fits, or `None` for one frame to a quarter of the run.

    Returns:
        displacement:
            The lags, the mean squared displacement, its exponent, and the diffusion coefficient.

    Raises:
        AnalysisError:
            The fit range holds fewer than 2 lags.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import mean_squared_displacement
        >>> walks = np.cumsum(np.random.default_rng(0).standard_normal((2000, 500)), axis=0)
        >>> displacement = mean_squared_displacement(walks)
        >>> round(displacement.exponent, 1), round(displacement.diffusion, 1)
        (1.0, 0.5)
    """
    array = _particle_axes(positions)
    msd = np.mean([tidynamics.msd(array[:, particle]) for particle in range(array.shape[1])], axis=0)[1:]
    all_lags = np.arange(1, array.shape[0], dtype=np.float64) * dt
    chosen = _fit_lags(all_lags, fit_range)
    exponent = float(np.polyfit(np.log(all_lags[chosen]), np.log(msd[chosen]), 1)[0])
    slope = float(np.polyfit(all_lags[chosen], msd[chosen], 1)[0])
    diffusion = slope / (2 * array.shape[2])
    if lags is None:
        return Displacement(all_lags, msd, exponent, diffusion)
    steps = np.asarray(lags, dtype=int)
    return Displacement(steps * dt, msd[steps - 1], exponent, diffusion)


@dataclass(frozen=True)
class Fluctuation:
    """Fluctuation of a series against the window size, and its scaling exponent.

    Args:
        scales:
            The window sizes.

        fluctuations:
            The fluctuation, or the rescaled range, at every window size.

        exponent:
            The slope of the log fluctuation against the log window size.
    """

    scales: Array
    fluctuations: Array
    exponent: float


@analysis(DataKind.SERIES)
def detrended_fluctuation(
    values: npt.ArrayLike,
    scales: npt.ArrayLike | None = None,
    order: int = 1,
    *,
    method: str = 'dfa',
) -> Fluctuation:
    """Return the fluctuation curve of a series by detrended fluctuation analysis or by the rescaled range.

    Detrended fluctuation analysis integrates the series, detrends overlapping windows with a polynomial, and fits the
    slope of the log fluctuation, 0.5 for white noise, the Hurst exponent `H` for fractional Gaussian noise, and `H + 1`
    for its integral, such as a random walk. The rescaled range gives `H` for noise.

    Args:
        values:
            The series.

        scales:
            The window sizes, or `None` for sizes spaced logarithmically up to a tenth of the series.

        order:
            The order of the detrending polynomial.

        method:
            `'dfa'` for detrended fluctuation analysis, or `'rs'` for the rescaled range.

    Returns:
        fluctuation:
            The window sizes, the fluctuations, and the exponent.

    Raises:
        AnalysisError:
            The series is shorter than 100 values, or the method is unknown.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import detrended_fluctuation
        >>> noise = np.random.default_rng(0).standard_normal(5000)
        >>> round(detrended_fluctuation(np.cumsum(noise)).exponent, 1)
        1.5
    """
    array = real_array(values, 'values')
    if array.size < MIN_FLUCTUATION_VALUES:
        error_msg = f'A fluctuation analysis needs at least {MIN_FLUCTUATION_VALUES} values, not {array.size}.'
        raise AnalysisError(error_msg)
    if method not in HURST_METHODS:
        error_msg = f'Unknown method {method!r}. Use one of {", ".join(HURST_METHODS)}.'
        raise AnalysisError(error_msg)
    sizes = None if scales is None else np.asarray(scales, dtype=int)
    if method == 'rs':
        exponent, (log_scales, log_values, _) = nolds.hurst_rs(array, nvals=sizes, fit='poly', debug_data=True)
    else:
        exponent, (log_scales, log_values, _) = nolds.dfa(
            array,
            nvals=sizes,
            order=order,
            fit_exp='poly',
            debug_data=True,
        )
    return Fluctuation(np.exp(log_scales), np.exp(log_values), float(exponent))


@analysis(DataKind.SERIES)
def hurst_exponent(
    values: npt.ArrayLike,
    scales: npt.ArrayLike | None = None,
    order: int = 1,
    *,
    method: str = 'dfa',
) -> float:
    """Return the Hurst exponent of a series by detrended fluctuation analysis or by the rescaled range.

    The exponent is 0.5 for uncorrelated noise, above 0.5 for persistent and below 0.5 for anti-persistent noise.
    Detrended fluctuation analysis gives `H + 1` for an integrated series, such as 1.5 for a random walk.

    Args:
        values:
            The series.

        scales:
            The window sizes, or `None` for sizes spaced logarithmically up to a tenth of the series.

        order:
            The order of the detrending polynomial.

        method:
            `'dfa'` for detrended fluctuation analysis, or `'rs'` for the rescaled range.

    Returns:
        hurst:
            The Hurst exponent.

    Raises:
        AnalysisError:
            The series is shorter than 100 values, or the method is unknown.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import hurst_exponent
        >>> round(hurst_exponent(np.random.default_rng(0).standard_normal(20000)), 1)
        0.5
    """
    return detrended_fluctuation(values, scales, order, method=method).exponent


def _reddening(segment: Array) -> float:
    """Return the ratio of the mean power of the lowest eighth of frequencies to that of the highest half."""
    power = np.abs(np.fft.rfft(segment)[1:]) ** 2
    low = power[: max(power.size // 8, 1)]
    high = power[power.size // 2 :]
    return float(low.mean() / high.mean()) if high.mean() > 0 else float('nan')


def _trend(indicator: Array) -> float:
    """Return Kendall's tau of an indicator against time over its defined values."""
    valid = ~np.isnan(indicator)
    return float(stats.kendalltau(np.flatnonzero(valid), indicator[valid]).statistic)


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

        skewness:
            The skewness in every window, `nan` before the first full window.

        reddening:
            The ratio of low- to high-frequency power in every window, which rises as fluctuations slow down.

        skewness_trend:
            Kendall's tau of the skewness against time.

        reddening_trend:
            Kendall's tau of the reddening against time.
    """

    variance: Array
    autocorrelation: Array
    variance_trend: float
    autocorrelation_trend: float
    skewness: Array
    reddening: Array
    skewness_trend: float
    reddening_trend: float


@analysis(DataKind.SERIES)
def early_warning(values: npt.ArrayLike, window: int, *, detrend: bool = True) -> EarlyWarning:
    """Return rolling variance, lag-1 autocorrelation, skewness, and spectral reddening before a tipping point.

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
        >>> from skcomplex.core.analysis import early_warning
        >>> rng = np.random.default_rng(0)
        >>> x = np.zeros(2000)
        >>> for t in range(1, 2000):
        ...     x[t] = (0.2 + 0.79 * t / 2000) * x[t - 1] + rng.standard_normal()
        >>> early_warning(x, 200).autocorrelation_trend > 0.5
        True
    """
    array = real_array(values, 'values')
    if not MIN_WINDOW <= window <= array.size:
        error_msg = f'window = {window} must lie between {MIN_WINDOW} and the length of the series, {array.size}.'
        raise AnalysisError(error_msg)
    indicators = np.full((4, array.size), np.nan)
    axis = np.arange(window)
    for end in range(window, array.size + 1):
        segment = array[end - window : end]
        if detrend:
            segment = segment - np.polyval(np.polyfit(axis, segment, 1), axis)
        indicators[:, end - 1] = (
            segment.var(),
            np.corrcoef(segment[:-1], segment[1:])[0, 1],
            stats.skew(segment),
            _reddening(segment),
        )
    variance, correlation, skewness, reddening = indicators
    return EarlyWarning(
        variance,
        correlation,
        _trend(variance),
        _trend(correlation),
        skewness,
        reddening,
        _trend(skewness),
        _trend(reddening),
    )
