"""Define observers, which receive the frames of a simulation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ..base import SimulationError
from ..compile import Problem, Vector
from ._frames import Frame
from ._trajectories import Trajectory


class Observer:
    """Base class of objects that receive the frames of a simulation.

    Subclasses override any of `on_start`, `on_frame`, and `on_stop`.
    """

    def on_start(self, problem: Problem) -> None:
        """Receive the compiled problem before the first frame.

        Args:
            problem:
                The compiled problem.
        """

    def on_frame(self, frame: Frame) -> None:
        """Receive one frame.

        Args:
            frame:
                The frame.
        """

    def on_stop(self) -> None:
        """Receive the end of the run."""


class Recorder(Observer):
    """Observer that keeps every frame it receives.

    Examples:
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import Population
        >>> from skcomplex.core.runtime import Recorder, Simulation
        >>> recorder = Recorder()
        >>> _ = Simulation(System(components=[Population('a')]), dt=1, observers=[recorder]).run(until=2)
        >>> len(recorder.trajectory)
        3
    """

    def __init__(self) -> None:
        self._problem: Problem | None = None
        self._times: list[float] = []
        self._states: list[Vector] = []
        self._parameters: list[Vector] = []
        self._frames: list[Frame] = []

    def on_start(self, problem: Problem) -> None:
        """Forget earlier frames and keep the problem.

        Args:
            problem:
                The compiled problem.
        """
        self._problem = problem
        self._times, self._states, self._parameters, self._frames = [], [], [], []

    def on_frame(self, frame: Frame) -> None:
        """Keep the time, state, and parameters of a frame.

        Args:
            frame:
                The frame.
        """
        self._times.append(frame.time)
        self._states.append(frame.state)
        self._parameters.append(frame.parameters)
        self._frames.append(frame)

    @property
    def trajectory(self) -> Trajectory:
        """Trajectory of the frames received so far."""
        if self._problem is None:
            error_msg = 'The recorder has not received a problem. Run a simulation that observes it first.'
            raise SimulationError(error_msg)
        n_frames = min(len(self._times), len(self._states), len(self._parameters), len(self._frames))
        return Trajectory(
            self._problem,
            np.asarray(self._times[:n_frames], dtype=float),
            np.asarray(self._states[:n_frames], dtype=float).reshape(n_frames, self._problem.states.size),
            np.asarray(self._parameters[:n_frames], dtype=float).reshape(n_frames, self._problem.parameters.size),
            agents=[frame.agents for frame in self._frames[:n_frames]],
            reporters=[frame.reporters for frame in self._frames[:n_frames]],
            observables=[frame.observables for frame in self._frames[:n_frames]],
            events=[event for frame in self._frames[:n_frames] for event in frame.events],
            structures=[frame.structures for frame in self._frames[:n_frames]],
        )
