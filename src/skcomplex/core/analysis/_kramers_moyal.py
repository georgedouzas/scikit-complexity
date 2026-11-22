"""Estimate the Kramers-Moyal drift and diffusion of a series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import convolve

from ..enums import DataKind
from ..errors import AnalysisError
from ..types import Array
from ._analysis_tags import AnalysisTags
from ._base import _read_series
from .types import DataSource

MIN_SERIES_SIZE = 2
ZERO_MASS = 1e-10
TAGS = AnalysisTags(DataKind.SERIES)


@dataclass(frozen=True)
class KramersMoyalResult:
    """A result of `kramers_moyal`, the drift and diffusion of one series.

    It holds the series, the centres of the bins of values, and the drift and the diffusion in every bin, such as a
    diffusion of 1 everywhere for a random walk whose steps have the variance `2 dt`. It reports the pull and the noise
    of wind speeds, heart rates, and exchange rates.

    Read more in the user guide, under Analyse.

    Args:
        values:
            The series.

        positions:
            The centres of the bins of values.

        drift:
            The first coefficient `<dx | x> / dt` in every bin.

        diffusion:
            The second coefficient `<dx² | x> / (2 dt)` in every bin, so `dx = a dt + sqrt(2 D) dW` gives `D`.

    See Also:
        kramers_moyal: Returns this result.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import kramers_moyal
        >>> x = np.cumsum(np.random.default_rng(0).normal(0, np.sqrt(0.02), 100_000))
        >>> kramers_moyal(x, dt=0.01, bins=21).positions.size
        21
    """

    values: Array
    positions: Array
    drift: Array
    diffusion: Array


def kramers_moyal(
    source: DataSource,
    *,
    variables: Sequence[str] | None = None,
    dt: float = 1.0,
    bins: int = 50,
    bandwidth: float | None = None,
) -> KramersMoyalResult:
    """Return the drift and the diffusion of one series.

    The drift is the mean step from a value per unit time and the diffusion half the mean squared step, so a series
    that follows `dx = a(x) dt + sqrt(2 D(x)) dW` gives `a` and `D`, such as a diffusion of 1 for a random walk whose
    steps have the variance `2 dt`. The steps are binned by the value they start from, and an Epanechnikov kernel
    smooths the bins, the estimator of Lamouroux and Lehnertz. It estimates the pull and the noise of wind speeds,
    heart rates, and exchange rates from one record.

    Read more in the user guide, under Analyse.

    Args:
        source:
            The trajectory, or an array that holds one series.

        variables:
            The variable of a trajectory to read, or `None` for the only scalar series.

        dt:
            The time between samples.

        bins:
            The number of bins of values.

        bandwidth:
            The width of the Epanechnikov kernel, or `None` for Silverman's rule.

    Returns:
        result:
            The series, the centres of the bins, and the drift and the diffusion in every bin.

    Raises:
        AnalysisError:
            A setting is not positive, the source does not hold one series of at least two values, or the series is
            constant without a bandwidth.

    See Also:
        sparse_identification: Finds the equations of a series from a library of terms.
        fit_law: Fits a law to the values of a series.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import kramers_moyal
        >>> x = np.cumsum(np.random.default_rng(0).normal(0, np.sqrt(0.02), 100_000))
        >>> round(float(np.median(kramers_moyal(x, dt=0.01, bins=21).diffusion)), 1)
        1.0
    """
    if dt <= 0 or bins < 1 or (bandwidth is not None and bandwidth <= 0):
        error_msg = f'kramers_moyal has dt={dt!r}, bins={bins!r}, and bandwidth={bandwidth!r}. Use positive values.'
        raise AnalysisError(error_msg)
    series = _read_series(source, variables)
    if series.values.shape[1] != 1:
        error_msg = (
            f'kramers_moyal reads one series, but the source has {series.values.shape[1]}: {", ".join(series.labels)}. '
            'Select one with variables.'
        )
        raise AnalysisError(error_msg)
    if series.values.shape[0] < MIN_SERIES_SIZE:
        error_msg = f'kramers_moyal reads increments, but the series has {series.values.shape[0]} value.'
        raise AnalysisError(error_msg)
    values = np.asarray(series.values[:, 0], dtype=float)
    width_of_kernel = _get_bandwidth(values, bandwidth)
    starts, increments = values[:-1], np.diff(values)
    edges = np.linspace(starts.min() - width_of_kernel, starts.max() + width_of_kernel, bins + 1)
    width = edges[1] - edges[0]
    histograms = np.stack([np.histogram(starts, edges, weights=increments**power)[0] for power in (0, 1, 2)])
    kernel = _get_kernel(width * np.arange(-bins - 1, bins + 2), width_of_kernel)
    mass, first, second = convolve(histograms, kernel[None, :], mode='same')
    counted = np.abs(mass) >= ZERO_MASS
    return KramersMoyalResult(
        values,
        edges[:-1] + width / 2,
        np.divide(first, mass, out=np.zeros(bins), where=counted) / dt,
        np.divide(second, 2 * mass, out=np.zeros(bins), where=counted) / dt,
    )


def _get_bandwidth(values: Array, bandwidth: float | None) -> float:
    """Return the given bandwidth, or the one of Silverman's rule."""
    if bandwidth is not None:
        return bandwidth
    deviation = float(np.std(values))
    if deviation == 0:
        error_msg = 'kramers_moyal cannot infer a bandwidth from a constant series. Give a bandwidth.'
        raise AnalysisError(error_msg)
    return (4 * deviation**5 / (3 * values.size)) ** (1 / 5)


def _get_kernel(points: Array, bandwidth: float) -> Array:
    """Return the Epanechnikov kernel of a bandwidth at some points."""
    scaled = points / bandwidth
    return np.where(np.abs(scaled) < 1, 0.75 * (1 - scaled**2) / bandwidth, 0.0)
