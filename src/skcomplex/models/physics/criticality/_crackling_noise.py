"""Measure the crackling-noise scaling of avalanches: the mean size against the duration and the collapse of shapes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import numpy as np

from ....analysis import EventAnalysis
from ....errors import AnalysisError
from ....types import Array

MIN_POINTS = 2
SHAPE_GRID = 50


class CracklingNoiseAnalysis(EventAnalysis):
    """Crackling-noise scaling of the avalanches of an activity series or of the relaxations of a lattice.

    It finds the avalanches as the event analysis of kind `'excursions'`, then fits the exponent `gamma` of the mean
    size against the duration, `<S>(T) ~ T^gamma`, predicts it as `(tau_T - 1) / (tau_S - 1)` from the exponents of
    the size and duration laws, such as those `FitAnalysis(law='power')` fits, and collapses the mean shapes of the
    chosen `durations`, which agree after scaling at a critical point.

    Args:
        variables:
            The one variable of a trajectory to read, or `None` for the only scalar series.

        threshold:
            The activity an avalanche exceeds at every step, subtracted from its profile.

        durations:
            The durations whose mean shapes are compared.

        min_count:
            The fewest avalanches of a duration used in the fit of the mean size.

        min_duration:
            The shortest duration used in the fit of the mean size.

        size_exponent:
            The exponent `tau_S` of the power law of the sizes, or `None` to leave out the prediction.

        duration_exponent:
            The exponent `tau_T` of the power law of the durations, or `None` to leave out the prediction.

    Attributes:
        exponent_:
            The exponent `gamma` of the mean size against the duration.

        predicted_:
            The exponent `(tau_T - 1) / (tau_S - 1)`, or `nan` without both exponents.

        mean_shapes_:
            The mean shape of every chosen duration, scaled to unit mean height, on a grid of 0 to 1.

        shape_spread_:
            The mean over the inner 80 percent of the scaled time of the relative difference between the largest and
            the smallest mean shape of the chosen durations, `nan` with fewer than 2 of them.

    Examples:
        >>> from skcomplex.models.physics.criticality import CracklingNoiseAnalysis
        >>> activity = [0, 1, 0, 1, 2, 1, 0, 1, 0, 2, 3, 2, 0]
        >>> crackling = CracklingNoiseAnalysis(min_count=1).check().prepare(activity).run()
        >>> round(crackling.exponent_, 2)
        1.55
    """

    def __init__(
        self,
        variables: Sequence[str] | None = None,
        *,
        threshold: float = 0.0,
        durations: Sequence[int] = (),
        min_count: int = 10,
        min_duration: int = 1,
        size_exponent: float | None = None,
        duration_exponent: float | None = None,
    ) -> None:
        super().__init__('excursions', variables, threshold=threshold)
        self.durations = durations
        self.min_count = min_count
        self.min_duration = min_duration
        self.size_exponent = size_exponent
        self.duration_exponent = duration_exponent

    def _check(self) -> Self:
        """Validate the fewest avalanches of a duration."""
        if self.min_count < 1:
            error_msg = f'{type(self).__name__} has min_count={self.min_count!r}. Use a positive value.'
            raise AnalysisError(error_msg)
        return self

    def _run(self, inputs: object = None) -> Self:
        """Find the avalanches, fit the mean size against the duration, predict its exponent, and compare shapes."""
        super()._run()
        distinct, counts = np.unique(self.durations_, return_counts=True)
        kept = distinct[(counts >= self.min_count) & (distinct >= max(self.min_duration, 1))]
        if kept.size < MIN_POINTS:
            error_msg = (
                f'Fewer than {MIN_POINTS} durations have {self.min_count} avalanches. Give more avalanches or lower '
                '`min_count`.'
            )
            raise AnalysisError(error_msg)
        means = np.asarray([self.sizes_[self.durations_ == duration].mean() for duration in kept])
        self.exponent_ = float(np.polyfit(np.log(kept), np.log(means), 1)[0])
        self.predicted_ = float('nan')
        if self.size_exponent is not None and self.duration_exponent is not None:
            self.predicted_ = float((self.duration_exponent - 1) / (self.size_exponent - 1))
        self.mean_shapes_ = {
            int(duration): self._mean_shape([shape for shape in self.shapes_ if shape.size == duration])
            for duration in self.durations
            if any(shape.size == duration for shape in self.shapes_)
        }
        self.shape_spread_ = float('nan')
        if len(self.mean_shapes_) >= MIN_POINTS:
            stacked = np.asarray(list(self.mean_shapes_.values()))
            inner = slice(SHAPE_GRID // 10, SHAPE_GRID - SHAPE_GRID // 10)
            spread = (stacked.max(axis=0) - stacked.min(axis=0))[inner] / stacked.mean(axis=0)[inner]
            self.shape_spread_ = float(np.mean(spread))
        return self

    @staticmethod
    def _mean_shape(shapes: list[Array]) -> Array:
        """Return the mean profile of avalanches of one duration, scaled by the mean height, on a grid of 0 to 1."""
        profiles = np.asarray(shapes, dtype=float)
        grid = np.linspace(0, 1, SHAPE_GRID)
        steps = np.linspace(0, 1, profiles.shape[1])
        mean = profiles.mean(axis=0)
        return np.interp(grid, steps, mean / mean.mean())
