"""Estimate the Kramers-Moyal drift and diffusion of a series as functions of its value."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import numpy as np

from ..errors import AnalysisError
from ..types import DataKind
from ._base import BaseAnalysis
from .types import Source


class KramersMoyalAnalysis(BaseAnalysis):
    """Kramers-Moyal drift and diffusion of one series, estimated with kernels from its increments by `kramersmoyal`.

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

    def _check(self) -> Self:
        """Validate the step, the number of bins, and the bandwidth."""
        if self.dt <= 0 or self.bins < 1 or (self.bandwidth is not None and self.bandwidth <= 0):
            error_msg = (
                f'{type(self).__name__} has dt={self.dt!r}, bins={self.bins!r}, and bandwidth={self.bandwidth!r}. '
                'Use positive values.'
            )
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the series kind."""
        return DataKind.SERIES

    def _prepare(self, source: Source) -> Self:
        """Read the one series of the source."""
        series = self._read_series(source, self.variables)
        if series.values.shape[1] != 1:
            error_msg = (
                f'{type(self).__name__} reads one series, but the source has {series.values.shape[1]}: '
                f'{", ".join(series.labels)}. Select one with variables.'
            )
            raise AnalysisError(error_msg)
        self.values_ = np.asarray(series.values[:, 0], dtype=float)
        return self

    def _run(self, inputs: object = None) -> Self:
        """Compute the drift and the diffusion in every bin."""
        moments, edges = self._import('kramersmoyal').km(
            self.values_,
            bins=np.asarray([self.bins]),
            powers=np.asarray([0, 1, 2]),
            bw=self.bandwidth,
        )
        self.positions_ = np.asarray(edges[0], dtype=float)
        self.drift_ = moments[1] / self.dt
        self.diffusion_ = moments[2] / self.dt
        return self
