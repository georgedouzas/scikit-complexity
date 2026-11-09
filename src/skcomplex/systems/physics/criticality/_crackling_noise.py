"""Measure the crackling-noise scaling of avalanches.

The scaling is the mean size against the duration and the collapse of shapes.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ....core.analysis import events
from ....core.analysis.types import Source
from ....core.errors import AnalysisError
from ....core.types import Array

MIN_POINTS = 2
SHAPE_GRID = 50


@dataclass(frozen=True)
class CracklingNoiseResult:
    """Crackling-noise scaling of avalanches.

    The avalanches are those of an activity series or the relaxations of a lattice.

    Args:
        times:
            The time of every value, or `None` for records.

        values:
            The series, as one column, or `None` for records.

        labels:
            The label of the series.

        sizes:
            The sum of the values of every avalanche.

        durations:
            The number of samples of every avalanche.

        shapes:
            The profile of every avalanche of a series, empty for records.

        exponent:
            The exponent `gamma` of the mean size against the duration.

        predicted:
            The exponent `(tau_T - 1) / (tau_S - 1)`, or `nan` without both exponents.

        mean_shapes:
            The mean shape of every chosen duration, scaled to unit mean height, on a grid of 0 to 1.

        shape_spread:
            The mean over the inner 80 percent of the scaled time of the relative difference between the largest and
            the smallest mean shape of the chosen durations, `nan` with fewer than 2 of them.
    """

    times: Array | None
    values: Array | None
    labels: list[str]
    sizes: Array
    durations: Array
    shapes: list[Array]
    exponent: float
    predicted: float
    mean_shapes: dict[int, Array]
    shape_spread: float


def crackling_noise(
    source: Source,
    *,
    variables: Sequence[str] | None = None,
    threshold: float = 0.0,
    durations: Sequence[int] = (),
    min_count: int = 10,
    min_duration: int = 1,
    size_exponent: float | None = None,
    duration_exponent: float | None = None,
) -> CracklingNoiseResult:
    """Return the crackling-noise scaling of avalanches.

    The avalanches are those of an activity series or the relaxations of a lattice.

    It finds the avalanches as the events of kind `'excursions'`, then fits the exponent `gamma` of the mean size
    against the duration, `<S>(T) ~ T^gamma`, predicts it as `(tau_T - 1) / (tau_S - 1)` from the exponents of the
    size and duration laws, such as those `fit_law(..., law='power')` fits, and collapses the mean shapes of the chosen
    `durations`, which agree after scaling at a critical point.

    Args:
        source:
            The trajectory, an array that holds the activity series, or records of avalanches.

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

    Returns:
        result:
            The avalanches, the exponent of the mean size and its prediction, and the mean shapes.

    Raises:
        AnalysisError:
            The fewest avalanches of a duration is not positive, or fewer than 2 durations have that many avalanches.

    Examples:
        >>> from skcomplex.systems.physics.criticality import crackling_noise
        >>> activity = [0, 1, 0, 1, 2, 1, 0, 1, 0, 2, 3, 2, 0]
        >>> round(crackling_noise(activity, min_count=1).exponent, 2)
        1.55
    """
    if min_count < 1:
        error_msg = f'crackling_noise has min_count={min_count!r}. Use a positive value.'
        raise AnalysisError(error_msg)
    found = events(source, kind='excursions', variables=variables, threshold=threshold)
    sizes = np.asarray(found.sizes, dtype=float)
    lasting = np.asarray(found.durations, dtype=float)
    shapes = list(found.shapes or [])
    distinct, counts = np.unique(lasting, return_counts=True)
    kept = distinct[(counts >= min_count) & (distinct >= max(min_duration, 1))]
    if kept.size < MIN_POINTS:
        error_msg = (
            f'Fewer than {MIN_POINTS} durations have {min_count} avalanches. Give more avalanches or lower '
            '`min_count`.'
        )
        raise AnalysisError(error_msg)
    means = np.asarray([sizes[lasting == duration].mean() for duration in kept])
    predicted = float('nan')
    if size_exponent is not None and duration_exponent is not None:
        predicted = float((duration_exponent - 1) / (size_exponent - 1))
    mean_shapes = {
        int(duration): _mean_shape([shape for shape in shapes if shape.size == duration])
        for duration in durations
        if any(shape.size == duration for shape in shapes)
    }
    shape_spread = float('nan')
    if len(mean_shapes) >= MIN_POINTS:
        stacked = np.asarray(list(mean_shapes.values()))
        inner = slice(SHAPE_GRID // 10, SHAPE_GRID - SHAPE_GRID // 10)
        spread = (stacked.max(axis=0) - stacked.min(axis=0))[inner] / stacked.mean(axis=0)[inner]
        shape_spread = float(np.mean(spread))
    return CracklingNoiseResult(
        found.times,
        found.values,
        found.labels,
        sizes,
        lasting,
        shapes,
        float(np.polyfit(np.log(kept), np.log(means), 1)[0]),
        predicted,
        mean_shapes,
        shape_spread,
    )


def _mean_shape(shapes: list[Array]) -> Array:
    """Return the mean profile of avalanches of one duration over the mean height.

    The profile lies on a grid of 0 to 1.
    """
    profiles = np.asarray(shapes, dtype=float)
    grid = np.linspace(0, 1, SHAPE_GRID)
    steps = np.linspace(0, 1, profiles.shape[1])
    mean = profiles.mean(axis=0)
    return np.interp(grid, steps, mean / mean.mean())
