"""Define the frames a simulation hands to its observers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..compile import Problem, Vector
from ..solvers import AgentSnapshot


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

        agents:
            A mapping from agent group path to the agents alive at this frame.

        reporters:
            A mapping from dotted reporter name to its value at this frame.
    """

    step: int
    time: float
    state: Vector
    parameters: Vector
    problem: Problem
    agents: Mapping[str, AgentSnapshot] = field(default_factory=dict)
    reporters: Mapping[str, float] = field(default_factory=dict)

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
        if name in self.reporters:
            return self.reporters[name]
        return self.problem.parameters.read(self.parameters, name)
