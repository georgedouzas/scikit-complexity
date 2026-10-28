"""Record the frames of a simulation as a trajectory."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ...core.errors import SimulationError
from ..problems import BaseProblem
from ._base import BaseObserver
from ._frames import Frame
from ._trajectories import Trajectory


class RecorderObserver(BaseObserver):
    """Observer that keeps every frame it receives.

    Attributes:
        problem_:
            The compiled problem of the run.

        frames_:
            The frames received so far.

    Examples:
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import PopulationElement
        >>> from skcomplex.engine.runtime import RecorderObserver, Simulation
        >>> recorder = RecorderObserver()
        >>> _ = Simulation(System(elements=[PopulationElement('a')]), dt=1, observers=[recorder]).run(until=2)
        >>> len(recorder.trajectory_.times)
        3
    """

    def _prepare(self, problem: BaseProblem) -> None:
        """Keep the problem and forget earlier frames."""
        self.problem_ = problem
        self.frames_: list[Frame] = []

    def _run(self, frame: Frame | None = None) -> None:
        """Keep a frame."""
        if frame is not None:
            self.frames_.append(frame)

    @property
    def trajectory_(self) -> Trajectory:
        """Trajectory of the frames received so far."""
        if not hasattr(self, 'problem_'):
            error_msg = 'The recorder has not received a problem. Run a simulation that observes it first.'
            raise SimulationError(error_msg)
        problem = self.problem_
        frames = list(self.frames_)
        n_frames = len(frames)
        return Trajectory(
            problem,
            np.asarray([frame.time for frame in frames], dtype=float),
            np.asarray([frame.state for frame in frames], dtype=float).reshape(n_frames, problem.states.size),
            np.asarray([frame.parameters for frame in frames], dtype=float).reshape(n_frames, problem.parameters.size),
            agents=[frame.agents for frame in frames],
            reporters=[frame.reporters for frame in frames],
            observables=[frame.observables for frame in frames],
            events=[event for frame in frames for event in frame.events],
            structures=[frame.structures for frame in frames],
        )
