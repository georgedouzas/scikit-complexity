"""Define the test of the speeds of agents or particles against the Maxwell-Boltzmann law."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
import numpy.typing as npt
from scipy import stats

from ....analysis import FitAnalysis
from ....engine.runtime import Trajectory
from ....errors import AnalysisError
from ._kinetic_temperature import KineticTemperatureAnalysis


class MaxwellBoltzmannAnalysis(FitAnalysis):
    """Kolmogorov-Smirnov test of the speeds of every frame against the Maxwell-Boltzmann law of their dimension.

    It is the fit of the named law `'maxwell'` to the speeds, generalized to `d` dimensions: the speeds of an ideal gas
    at temperature `T` follow a chi law of `d` degrees of freedom and scale `sqrt(T / m)`, the Maxwell law in three
    dimensions and the Rayleigh law in two. The temperature is the kinetic temperature of every frame unless it is
    given. It reads the velocities of the agents or particles of a trajectory, or an array of velocities of one
    snapshot, one row per agent, or of one snapshot per frame along its first axis.

    Args:
        group:
            The path of the agent or particle group of a trajectory, or `None` for the first one.

        mass:
            The mass of every agent.

        temperature:
            The temperature of the law, or `None` for the kinetic temperature of every frame.

    Attributes:
        values_:
            The velocities, frames by agents by dimensions.

        temperature_:
            The temperature of the tested law at every frame.

        statistic_:
            The Kolmogorov-Smirnov statistic of every frame.

        p_value_:
            The p-value of every frame.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.models.physics.statistical_mechanics import MaxwellBoltzmannAnalysis
        >>> velocities = np.random.default_rng(0).standard_normal((2000, 3))
        >>> bool(MaxwellBoltzmannAnalysis().check().prepare(velocities).run().p_value_[0] > 0.05)
        True
    """

    def __init__(self, group: str | None = None, *, mass: float = 1.0, temperature: float | None = None) -> None:
        super().__init__("maxwell", group=group)
        self.mass = mass
        self.temperature = temperature

    def _check(self) -> None:
        """Validate the mass and the temperature."""
        if self.mass <= 0 or (self.temperature is not None and self.temperature <= 0):
            error_msg = (
                f"{type(self).__name__} has mass={self.mass!r} and temperature={self.temperature!r}. Use positive "
                "values."
            )
            raise AnalysisError(error_msg)

    def _prepare(self, source: object) -> None:
        """Read the velocities of the agents of every frame from a trajectory or an array."""
        if not isinstance(source, Trajectory | np.ndarray | list | tuple):
            error_msg = f'{type(self).__name__} reads a trajectory or an array of velocities, not {source!r}.'
            raise AnalysisError(error_msg)
        self.values_ = self._read_velocities(source, self.group)

    def _run(self, inputs: object = None) -> None:
        """Test the speeds of every frame."""
        tests = [self._maxwell_boltzmann(frame, self.mass, self.temperature) for frame in self.values_]
        self.temperature_, self.statistic_, self.p_value_ = np.asarray(tests).T

    @staticmethod
    def _maxwell_boltzmann(frame: npt.ArrayLike, mass: float, temperature: float | None) -> tuple[float, float, float]:
        """Return the temperature and the Kolmogorov-Smirnov test of the speeds of one frame against the chi law."""
        velocities = np.asarray(frame, dtype=float)
        chosen = (
            float(KineticTemperatureAnalysis._kinetic_temperature(velocities, mass))
            if temperature is None
            else temperature
        )
        speeds = np.linalg.norm(velocities, axis=-1)
        law = stats.chi(df=velocities.shape[-1], scale=math.sqrt(chosen / mass))
        test = stats.kstest(speeds, law.cdf)
        return chosen, float(test.statistic), float(test.pvalue)
