"""Estimate the Kramers-Moyal drift and diffusion of a series as functions of its value."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import convolve

from ..errors import AnalysisError
from ..types import Array
from ._base import _read_series
from .types import Source

MIN_SERIES_SIZE = 2

ZERO_MASS = 1e-10


@dataclass(frozen=True)
class KramersMoyalResult:
    """Drift and diffusion of one series, estimated from its increments.

    Args:
        values:
            The series.

        positions:
            The centres of the bins of values.

        drift:
            The first coefficient `<dx | x> / dt` in every bin.

        diffusion:
            The second coefficient `<dx² | x> / (2 dt)` in every bin, so `dx = a dt + sqrt(2 D) dW` gives `D`.
    """

    values: Array
    positions: Array
    drift: Array
    diffusion: Array


def kramers_moyal(
    source: Source,
    *,
    variables: Sequence[str] | None = None,
    dt: float = 1.0,
    bins: int = 50,
    bandwidth: float | None = None,
) -> KramersMoyalResult:
    """Return the drift and the diffusion of one series, estimated from the moments of its increments.

    The increments are binned by the value they start from, and an Epanechnikov kernel smooths the bins. This is the
    estimator of Lamouroux and Lehnertz.

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

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import kramers_moyal
        >>> rng = np.random.default_rng(0)
        >>> x = np.zeros(100_000)
        >>> for t in range(1, x.size):
        ...     x[t] = x[t - 1] - x[t - 1] * 0.01 + np.sqrt(0.02) * rng.standard_normal()
        >>> round(float(kramers_moyal(x, dt=0.01, bins=21).diffusion[10]), 1)
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
