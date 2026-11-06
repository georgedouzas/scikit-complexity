"""Record the frames of a simulation as a trajectory."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt

from ...errors import SimulationError, ValidationError
from ..problems import BaseProblemSpec, FrameSpec
from ._base import BaseObserver
from ._trajectories import Trajectory


class RecorderObserver(BaseObserver):
    """Observer that keeps every frame it receives.

    Attributes:
        problem_:
            The compiled problem of the run.

        frames_:
            The frames received so far.

    Examples:
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems.biology.common.populations import PopulationElement
        >>> from skcomplex.core.engine.runtime import RecorderObserver, Simulation
        >>> recorder = RecorderObserver()
        >>> simulation = Simulation(System(elements=[PopulationElement('a')]), dt=1, observers=[recorder])
        >>> _ = simulation.prepare().run(until=2)
        >>> len(recorder.trajectory_.times)
        3
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

    def _get_parameters(self, frames: list[FrameSpec]) -> npt.NDArray[np.float64]:
        """Return the parameters of the frames, a read-only view of one row when every frame shares the same array."""
        shape = (len(frames), self.problem_.parameters.size)
        if frames and all(frame.parameters is frames[0].parameters for frame in frames):
            return np.broadcast_to(np.asarray(frames[0].parameters, dtype=float), shape)
        return np.asarray([frame.parameters for frame in frames], dtype=float).reshape(shape)

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
            self._get_parameters(frames),
            agents=[frame.agents for frame in frames],
            reporters=[frame.reporters for frame in frames],
            observables=[frame.observables for frame in frames],
            events=[event for frame in frames for event in frame.events],
            structures=[frame.structures for frame in frames],
        )
