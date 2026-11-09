"""Define the susceptibility of an order parameter from its fluctuations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ....core.analysis import distribution
from ....core.analysis.types import Source
from ....core.errors import AnalysisError
from ....core.types import Array


@dataclass(frozen=True)
class SusceptibilityResult:
    """Susceptibility of every series of samples of an order parameter.

    There is one series per control value.

    Args:
        values:
            The samples, one per column.

        labels:
            The label of every series.

        mean:
            The mean absolute order parameter of every series.

        variance:
            The variance of the absolute order parameter of every series.

        skewness:
            The skewness of the absolute order parameter of every series.

        kurtosis:
            The excess kurtosis of the absolute order parameter of every series.

        susceptibility:
            The susceptibility of every series.

        peak:
            The control parameter value of the largest susceptibility, refined by a parabola through its neighbours, or
            `None` without control parameter values.
    """

    values: Array
    labels: list[str]
    mean: Array
    variance: Array
    skewness: Array
    kurtosis: Array
    susceptibility: Array
    peak: float | None


def susceptibility(
    source: Source,
    *,
    variables: Sequence[str] | None = None,
    group: str | None = None,
    size: float = 1.0,
    temperatures: Sequence[float] | None = None,
    parameters: Sequence[float] | None = None,
) -> SusceptibilityResult:
    """Return the susceptibility `N (<m²> - <|m|>²)` of every series of samples.

    The samples are those of an order parameter.

    It is the distribution of kind `'moments'` of the absolute samples, whose variance it scales by the system size `N`
    and divides by the temperature when `temperatures` are given. Over a sweep of a control parameter, one series per
    control value, the susceptibility peaks near the transition, and the peak is refined by a parabola through its
    neighbours.

    Args:
        source:
            The trajectory, or an array with one column per series of samples.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series.

        group:
            The path of the agent or particle group whose speeds are read, or `None` for the series of a trajectory.

        size:
            The system size `N` that scales the variance.

        temperatures:
            The temperature of every series that divides the susceptibility, or `None` to leave it undivided.

        parameters:
            The control parameter value of every series, or `None` without a peak.

    Returns:
        result:
            The moments of the absolute samples, the susceptibility, and its peak.

    Raises:
        AnalysisError:
            The system size is not positive.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import susceptibility
        >>> susceptibility([0.5, -1.0, 0.5, -1.0], size=4).susceptibility.tolist()
        [0.25]
    """
    if size <= 0:
        error_msg = f"susceptibility has size={size!r}. Use a positive size."
        raise AnalysisError(error_msg)
    moments = distribution(source, kind="moments", variables=variables, group=group, absolute=True)
    if moments.mean is None or moments.variance is None or moments.skewness is None or moments.kurtosis is None:
        error_msg = "The distribution of kind moments returned no moments."
        raise AnalysisError(error_msg)
    values = size * moments.variance
    if temperatures is not None:
        values = values / np.asarray(temperatures, dtype=float)
    peak = None if parameters is None else _refined_peak(np.asarray(parameters, dtype=float), values)
    return SusceptibilityResult(
        moments.values,
        moments.labels,
        moments.mean,
        moments.variance,
        moments.skewness,
        moments.kurtosis,
        values,
        peak,
    )


def _refined_peak(grid: Array, values: Array) -> float:
    """Return the abscissa of the vertex of the parabola through the largest value.

    The parabola also runs through the neighbours of the largest value.
    """
    index = int(np.argmax(values))
    if index in (0, values.size - 1):
        return float(grid[index])
    coefficients = np.polyfit(grid[index - 1 : index + 2], values[index - 1 : index + 2], 2)
    if coefficients[0] >= 0:
        return float(grid[index])
    return float(-coefficients[1] / (2 * coefficients[0]))
