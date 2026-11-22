"""Record the frames of a simulation as a trajectory."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

import numpy as np
import numpy.typing as npt

from ...errors import SimulationError, ValidationError
from ..problems import BaseProblemSpec, FrameSpec
from ._base import BaseObserver
from ._trajectories import Trajectory


class RecorderObserver(BaseObserver):
    """An observer that keeps every frame it receives.

    It stores each frame in `frames_`, such as the frames of times 0, 1, and 2 of a run until 2, and gathers them into
    `trajectory_`. It keeps the course of an epidemic, the prices of a market, and the moves of a flock for later
    reading.

    Read more in the user guide, under Observe.

    Attributes:
        problem_:
            The compiled problem of the run.

        frames_:
            The frames received so far.

        trajectory_:
            The trajectory of the frames received so far.

    See Also:
        BaseObserver: The base of every observer.
        Trajectory: The record of a run that `simulate` returns.

    Examples:
        >>> from skcomplex.core.engine.runtime import RecorderObserver
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> recorder = RecorderObserver()
        >>> _ = LogisticGrowthSystem().prepare().simulate(until=2, dt=1, observers=[recorder])
        >>> [frame.time for frame in recorder.frames_]
        [0.0, 1.0, 2.0]
    """

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Keep the problem and forget earlier frames.

        Args:
            context:
                The compiled problem the run advances.

        Raises:
            ValidationError:
                No problem is passed.
        """
        if context is None:
            error_msg = 'A recorder prepares on a compiled problem. Pass the problem to `prepare`.'
            raise ValidationError(error_msg)
        self.problem_ = context
        self.frames_: list[FrameSpec] = []

    def _run(self, frame: FrameSpec | None = None) -> None:
        """Keep a frame."""
        if frame is not None:
            self.frames_.append(frame)

    def _get_relaid(self, frame: FrameSpec) -> FrameSpec:
        """Return a frame laid out like the last frame.

        A frame from before a group grew holds fewer rows, and the rows it lacks are free, as unused rows are.
        """
        problem = self.frames_[-1].problem
        if frame.problem is problem:
            return frame
        state = problem.states.get_relaid(frame.state, frame.problem.states, problem.initial_state)
        parameters = problem.parameters.get_relaid(
            frame.parameters,
            frame.problem.parameters,
            problem.initial_parameters,
        )
        return replace(frame, state=state, parameters=parameters, problem=problem)

    def _get_parameters(self, frames: list[FrameSpec], problem: BaseProblemSpec) -> npt.NDArray[np.float64]:
        """Return the parameters of the frames.

        They are a read-only view of one row when every frame shares the same array.
        """
        shape = (len(frames), problem.parameters.size)
        if frames and all(frame.parameters is frames[0].parameters for frame in frames):
            return np.broadcast_to(np.asarray(frames[0].parameters, dtype=float), shape)
        return np.asarray([frame.parameters for frame in frames], dtype=float).reshape(shape)

    @property
    def trajectory_(self) -> Trajectory:
        """Trajectory of the frames received so far."""
        if not hasattr(self, 'problem_'):
            error_msg = 'The recorder has not received a problem. Run a simulation that observes it first.'
            raise SimulationError(error_msg)
        frames = [self._get_relaid(frame) for frame in self.frames_]
        problem = frames[-1].problem if frames else self.problem_
        n_frames = len(frames)
        return Trajectory(
            problem,
            np.asarray([frame.time for frame in frames], dtype=float),
            np.asarray([frame.state for frame in frames], dtype=float).reshape(n_frames, problem.states.size),
            self._get_parameters(frames, problem),
            agents=[frame.agents for frame in frames],
            reporters=[frame.reporters for frame in frames],
            observables=[frame.observables for frame in frames],
            events=[event for frame in frames for event in frame.events],
            structures=[frame.structures for frame in frames],
        )
