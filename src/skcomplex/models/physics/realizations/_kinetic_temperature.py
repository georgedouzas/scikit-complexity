"""Define the kinetic temperature of moving agents or particles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

import numpy as np
import numpy.typing as npt

from ....analysis import DistributionAnalysis
from ....core.errors import AnalysisError
from ....engine.runtime import Trajectory

type Array = npt.NDArray[np.float64]


class KineticTemperatureAnalysis(DistributionAnalysis):
    """Kinetic temperature `m <|v|²> / d` of the agents of every frame in `d` dimensions, with `k_B = 1`.

    It is the second moment of the speeds of the agents, the distribution analysis of kind `'moments'`, scaled by the
    mass over the dimension, as equipartition gives `k_B T / 2` per degree of freedom. It reads the velocities of the
    agents or particles of a trajectory, or an array of velocities of one snapshot, one row per agent, or of one
    snapshot per frame along its first axis.

    Args:
        group:
            The path of the agent or particle group of a trajectory, or `None` for the first one.

        mass:
            The mass of every agent.

    Attributes:
        values_:
            The velocities, frames by agents by dimensions.

        temperature_:
            The kinetic temperature of every frame.

    Examples:
        >>> from skcomplex.models.physics.realizations import KineticTemperatureAnalysis
        >>> KineticTemperatureAnalysis().check().prepare([[1.0, 0.0], [0.0, 1.0]]).run().temperature_
        array([0.5])
    """

    def __init__(self, group: str | None = None, *, mass: float = 1.0) -> None:
        super().__init__("moments", group=group)
        self.mass = mass

    def _check(self) -> Self:
        """Validate the mass."""
        if self.mass <= 0:
            error_msg = f"{type(self).__name__} has mass={self.mass!r}. Use a positive mass."
            raise AnalysisError(error_msg)
        return self

    def _prepare(self, source: Trajectory | npt.ArrayLike) -> Self:
        """Read the velocities of the agents of every frame."""
        self.values_ = self._read_velocities(source, self.group)
        return self

    def _run(self) -> Self:
        """Compute the kinetic temperature of every frame."""
        self.temperature_ = self._kinetic_temperature(self.values_, self.mass)
        return self

    @staticmethod
    def _kinetic_temperature(velocities: Array, mass: float) -> Array:
        """Return `m <|v|²> / d` of the velocities of the agents of every frame in `d` dimensions."""
        return mass * np.mean(np.sum(velocities**2, axis=-1), axis=-1) / velocities.shape[-1]
