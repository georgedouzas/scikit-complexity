"""Define the frames a simulation hands to its observers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..compile import Problem, Vector


@dataclass(frozen=True)
class Frame:
    """State of a simulation at one point in time.

    Args:
        step:
            The index of the frame, starting at 0.

        time:
            The simulated time.

        state:
            A copy of the state vector.

        parameters:
            A copy of the parameter vector.

        problem:
            The compiled problem, which gives the names of the vector components.
    """

    step: int
    time: float
    state: Vector
    parameters: Vector
    problem: Problem

    def value(self, name: str) -> npt.NDArray[np.float64] | float:
        """Return the value of a state or a parameter by dotted name.

        Args:
            name:
                The dotted name, such as `rabbits.n` or `growth.r`.

        Returns:
            value:
                A number for a scalar and an array for a vector.
        """
        if name in self.problem.states:
            return self.problem.states.read(self.state, name)
        return self.problem.parameters.read(self.parameters, name)
