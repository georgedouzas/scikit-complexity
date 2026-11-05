"""Estimate the Kramers-Moyal drift and diffusion of a series as functions of its value."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
from scipy.signal import convolve

from ..errors import AnalysisError
from ..types import Array, DataKind
from ._base import BaseAnalysis
from .types import Source

MIN_SERIES_SIZE = 2

ZERO_MASS = 1e-10


class KramersMoyalAnalysis(BaseAnalysis):
    """Kramers-Moyal drift and diffusion of one series, estimated with kernels from its increments.

    The increments are binned by the value they start from, and an Epanechnikov kernel smooths the bins. This is the
    estimator of Lamouroux and Lehnertz.

    Args:
        variables:
            The variable of a trajectory to read, or `None` for the only scalar series.

        dt:
            The time between samples.

        bins:
            The number of bins of values.

        bandwidth:
            The width of the Epanechnikov kernel, or `None` for Silverman's rule.

    Attributes:
        values_:
            The series.

        positions_:
            The centres of the bins of values.

        drift_:
            The first coefficient `<dx | x> / dt` in every bin.

        diffusion_:
            The second coefficient `<dx² | x> / (2 dt)` in every bin, so `dx = a dt + sqrt(2 D) dW` gives `D`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import KramersMoyalAnalysis
        >>> rng = np.random.default_rng(0)
        >>> x = np.zeros(100_000)
        >>> for t in range(1, x.size):
        ...     x[t] = x[t - 1] - x[t - 1] * 0.01 + np.sqrt(0.02) * rng.standard_normal()
        >>> analysis = KramersMoyalAnalysis(dt=0.01, bins=21).check().prepare(x).run()
        >>> round(float(analysis.diffusion_[10]), 1)
        1.0
    """

    def __init__(
        self,
        variables: Sequence[str] | None = None,
        *,
        dt: float = 1.0,
        bins: int = 50,
        bandwidth: float | None = None,
    ) -> None:
        self.variables = variables
        self.dt = dt
        self.bins = bins
        self.bandwidth = bandwidth

    def _check(self) -> None:
        """Validate the step, the number of bins, and the bandwidth."""
        if self.dt <= 0 or self.bins < 1 or (self.bandwidth is not None and self.bandwidth <= 0):
            error_msg = (
                f'{type(self).__name__} has dt={self.dt!r}, bins={self.bins!r}, and bandwidth={self.bandwidth!r}. '
                'Use positive values.'
            )
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the series kind."""
        return DataKind.SERIES

    def _prepare(self, source: Source) -> None:
        """Read the one series of the source."""
        series = self._read_series(source, self.variables)
        if series.values.shape[1] != 1:
            error_msg = (
                f'{type(self).__name__} reads one series, but the source has {series.values.shape[1]}: '
                f'{", ".join(series.labels)}. Select one with variables.'
            )
            raise AnalysisError(error_msg)
        if series.values.shape[0] < MIN_SERIES_SIZE:
            error_msg = f'{type(self).__name__} reads increments, but the series has {series.values.shape[0]} value.'
            raise AnalysisError(error_msg)
        self.values_ = np.asarray(series.values[:, 0], dtype=float)

    def _run(self, inputs: object = None) -> None:
        """Compute the drift and the diffusion in every bin."""
        bandwidth = self._get_bandwidth()
        starts, increments = self.values_[:-1], np.diff(self.values_)
        edges = np.linspace(starts.min() - bandwidth, starts.max() + bandwidth, self.bins + 1)
        width = edges[1] - edges[0]
        histograms = np.stack([np.histogram(starts, edges, weights=increments**power)[0] for power in (0, 1, 2)])
        kernel = self._get_kernel(width * np.arange(-self.bins - 1, self.bins + 2), bandwidth)
        mass, first, second = convolve(histograms, kernel[None, :], mode='same')
        counted = np.abs(mass) >= ZERO_MASS
        self.positions_ = edges[:-1] + width / 2
        self.drift_ = np.divide(first, mass, out=np.zeros(self.bins), where=counted) / self.dt
        self.diffusion_ = np.divide(second, 2 * mass, out=np.zeros(self.bins), where=counted) / self.dt

    def _get_bandwidth(self) -> float:
        """Return the given bandwidth, or the one of Silverman's rule."""
        if self.bandwidth is not None:
            return self.bandwidth
        deviation = float(np.std(self.values_))
        if deviation == 0:
            error_msg = f'{type(self).__name__} cannot infer a bandwidth from a constant series. Give a bandwidth.'
            raise AnalysisError(error_msg)
        return (4 * deviation**5 / (3 * self.values_.size)) ** (1 / 5)

    @staticmethod
    def _get_kernel(points: Array, bandwidth: float) -> Array:
        """Return the Epanechnikov kernel of a bandwidth at some points."""
        scaled = points / bandwidth
        return np.where(np.abs(scaled) < 1, 0.75 * (1 - scaled**2) / bandwidth, 0.0)
