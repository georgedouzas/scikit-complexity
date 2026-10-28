"""Optimal control of a run over its horizon, sampled on a grid of times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ...core.types import Vector
from ..problems import BaseValue


@dataclass(frozen=True)
class ControlSolution(BaseValue):
    """Optimal control of a run over its horizon, sampled on a grid of times.

    Args:
        times:
            The times of the grid, from the start to the horizon.

        states:
            The state vector at every time of the grid, with the controls at their optimal values, one row per time.

        costates:
            The costate of every state the controls steer, by state name, one value per time, or empty when the
            solver does not compute costates.

        cost:
            The value of the objective, the running value integrated over the run plus the terminal value.

        iterations:
            The number of sweeps or optimiser iterations the solver took.
    """

    times: Vector
    states: Vector
    costates: Mapping[str, Vector]
    cost: float
    iterations: int

    def state_at(self, t: float) -> Vector:
        """Return the state vector at a time of the run, interpolated linearly between the times of the grid.

        Args:
            t:
                The time.

        Returns:
            state:
                The state vector.
        """
        return np.array([np.interp(t, self.times, column) for column in self.states.T])
