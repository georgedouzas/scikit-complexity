"""Record the frames of a simulation as a trajectory."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt

from ..compile import Problem


class Trajectory:
    """Recorded states and parameters of a simulation over time.

    Args:
        problem:
            The compiled problem, which gives the names of the vector components.

        times:
            The time of each recorded frame.

        states:
            One state vector per row.

        parameters:
            One parameter vector per row.

    Examples:
        >>> from skcomplex.model import System
        >>> from skcomplex.models.ecology import ExponentialGrowth, Population
        >>> from skcomplex.runtime import Simulation
        >>> system = System(elements=[Population('a', n=1)], interactions=[ExponentialGrowth('g', target='a', r=1)])
        >>> trajectory = Simulation(system, dt=0.5).run(until=1).trajectory_
        >>> trajectory.times.tolist()
        [0.0, 0.5, 1.0]
        >>> trajectory['a.n'].round(3).tolist()
        [1.0, 1.649, 2.718]
    """

    def __init__(
        self,
        problem: Problem,
        times: npt.NDArray[np.float64],
        states: npt.NDArray[np.float64],
        parameters: npt.NDArray[np.float64],
    ) -> None:
        self.problem = problem
        self.times = times
        self.states = states
        self.parameters = parameters

    def __len__(self) -> int:
        """Return the number of recorded frames."""
        return len(self.times)

    def __getitem__(self, name: str) -> npt.NDArray[np.float64]:
        """Return the values of a state or a parameter over time by dotted name.

        Args:
            name:
                The dotted name, such as `rabbits.n` or `growth.r`.

        Returns:
            values:
                One value per frame for a scalar, or one row per frame for a vector.

        Raises:
            KeyError:
                The name is neither a state nor a parameter.
        """
        if name in self.problem.states:
            return np.asarray(self.problem.states.read(self.states, name))
        if name in self.problem.parameters:
            return np.asarray(self.problem.parameters.read(self.parameters, name))
        names = [*self.problem.states.names, *self.problem.parameters.names]
        error_msg = f'No state or parameter named {name!r}. Available names are: {", ".join(names)}.'
        raise KeyError(error_msg)
