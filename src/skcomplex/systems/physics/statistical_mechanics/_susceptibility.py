"""Define the susceptibility of an order parameter from its fluctuations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ....core.analysis import DistributionAnalysis
from ....core.errors import AnalysisError
from ....core.types import Array


class SusceptibilityAnalysis(DistributionAnalysis):
    """Susceptibility `N (<m²> - <|m|>²)` of every series of samples of an order parameter, one per control value.

    It is the distribution analysis of kind `'moments'` of the absolute samples, whose variance it scales by the system
    size `N` and divides by the temperature when `temperatures` are given. Over a sweep of a control parameter the
    susceptibility peaks near the transition, and the peak is refined by a parabola through its neighbours.

    Args:
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

    Attributes:
        values_:
            The samples, one per column.

        labels_:
            The label of every series.

        mean_:
            The mean absolute order parameter of every series.

        variance_:
            The variance of the absolute order parameter of every series.

        skewness_:
            The skewness of the absolute order parameter of every series.

        kurtosis_:
            The excess kurtosis of the absolute order parameter of every series.

        susceptibility_:
            The susceptibility of every series.

        peak_:
            The control parameter value of the largest susceptibility, refined by a parabola through its neighbours, or
            `None` without control parameter values.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import SusceptibilityAnalysis
        >>> susceptibility = SusceptibilityAnalysis(size=4).prepare([0.5, -1.0, 0.5, -1.0]).run()
        >>> susceptibility.susceptibility_.tolist()
        [0.25]
    """

    def __init__(
        self,
        variables: Sequence[str] | None = None,
        *,
        group: str | None = None,
        size: float = 1.0,
        temperatures: Sequence[float] | None = None,
        parameters: Sequence[float] | None = None,
    ) -> None:
        super().__init__("moments", variables, group=group, absolute=True)
        self.size = size
        self.temperatures = temperatures
        self.parameters = parameters

    def _check(self) -> None:
        """Validate the system size."""
        if self.size <= 0:
            error_msg = f"{type(self).__name__} has size={self.size!r}. Use a positive size."
            raise AnalysisError(error_msg)

    def _run(self, inputs: object = None) -> None:
        """Compute the moments of the absolute samples, the susceptibility, and its peak."""
        super()._run()
        self.susceptibility_ = self.size * self.variance_
        if self.temperatures is not None:
            self.susceptibility_ = self.susceptibility_ / np.asarray(self.temperatures, dtype=float)
        self.peak_ = (
            None
            if self.parameters is None
            else self._refined_peak(np.asarray(self.parameters, dtype=float), self.susceptibility_)
        )

    @staticmethod
    def _refined_peak(grid: Array, values: Array) -> float:
        """Return the abscissa of the vertex of the parabola through the largest value and its neighbours."""
        index = int(np.argmax(values))
        if index in (0, values.size - 1):
            return float(grid[index])
        coefficients = np.polyfit(grid[index - 1 : index + 2], values[index - 1 : index + 2], 2)
        if coefficients[0] >= 0:
            return float(grid[index])
        return float(-coefficients[1] / (2 * coefficients[0]))
