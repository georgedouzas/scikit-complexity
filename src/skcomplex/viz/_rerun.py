"""Stream simulations to the Rerun viewer or to a recording file."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from pathlib import Path

import numpy as np
import rerun as rr

from ..compile import CompiledElement, Problem
from ..runtime import Frame, Observer
from ._visuals import Visual, resolve_visual


class RerunObserver(Observer):
    """Observer that logs frames to Rerun, with entity paths that follow the system tree.

    Each logged frame sets the `sim_time` and `step` timelines, logs every element with the visual its type
    declares, and logs every parameter under `parameters/<name>`.

    Args:
        application_id:
            The Rerun application id.

        spawn:
            Whether to spawn a Rerun viewer.

        connect:
            Whether to connect to a running Rerun viewer.

        save_path:
            The path of a `.rrd` file to save the recording to.

        interval:
            The number of frames between logged frames.

    Attributes:
        recording_:
            The Rerun recording stream.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from skcomplex.model import System
        >>> from skcomplex.models.ecology import ExponentialGrowth, Population
        >>> from skcomplex.runtime import Simulation
        >>> from skcomplex.viz import RerunObserver
        >>> system = System(elements=[Population('a', n=1)], interactions=[ExponentialGrowth('g', target='a', r=1)])
        >>> path = Path(tempfile.mkdtemp()) / 'growth.rrd'
        >>> _ = Simulation(system, dt=0.1, observers=[RerunObserver('growth', save_path=path)]).run(until=1)
        >>> path.exists()
        True
    """

    def __init__(
        self,
        application_id: str = 'skcomplex',
        *,
        spawn: bool = False,
        connect: bool = False,
        save_path: str | Path | None = None,
        interval: int = 1,
    ) -> None:
        self.application_id = application_id
        self.spawn = spawn
        self.connect = connect
        self.save_path = save_path
        self.interval = interval

    def on_start(self, problem: Problem) -> None:
        """Create the recording and send it to its sinks.

        Args:
            problem:
                The compiled problem.
        """
        self.recording_ = rr.RecordingStream(self.application_id)
        if self.save_path is not None:
            self.recording_.save(self.save_path)
        if self.spawn:
            self.recording_.spawn()
        elif self.connect:
            self.recording_.connect_grpc()
        self._visuals: list[tuple[CompiledElement, Visual]] = [
            (element, resolve_visual(type(element.element).visual)) for element in problem.elements
        ]
        self.logged_frames_ = 0

    def on_frame(self, frame: Frame) -> None:
        """Log a frame when its step is a multiple of the interval.

        Args:
            frame:
                The frame.
        """
        if frame.step % self.interval:
            return
        self.recording_.set_time('step', sequence=frame.step)
        self.recording_.set_time('sim_time', duration=frame.time)
        for element, visual in self._visuals:
            visual.log(self.recording_, element, frame)
        for name in frame.problem.parameters.names:
            self.recording_.log(f'parameters/{name}', rr.Scalars(np.atleast_1d(frame.value(name))))
        self.logged_frames_ += 1

    def on_stop(self) -> None:
        """Flush the recording."""
        self.recording_.flush()
