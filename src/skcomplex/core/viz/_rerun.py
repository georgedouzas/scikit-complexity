"""Stream simulations to the Rerun viewer or to a recording file."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import sys
from pathlib import Path

import numpy as np
import rerun as rr

from ..compile import CompiledElement, Problem
from ..model import NetworkSpace
from ..runtime import Frame, Observer
from ._visuals import AgentsVisual, Visual, resolve_visual


def _viewer_executable() -> str | None:
    """Return the Rerun viewer installed next to the running interpreter, or `None` to search `PATH`."""
    for name in ('rerun', 'rerun.exe'):
        candidate = Path(sys.executable).parent / name
        if candidate.is_file():
            return str(candidate)
    return None


class RerunObserver(Observer):
    """Observer that logs frames to Rerun, with entity paths that follow the system tree.

    Each logged frame sets the `sim_time` and `step` timelines, logs every element with the visual its type
    declares, and logs every parameter under `parameters/<name>`.

    Args:
        application_id:
            The Rerun application id.

        spawn:
            Whether to spawn the Rerun viewer installed with the running interpreter, or the one on `PATH`.

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
        >>> from skcomplex.core.model import System
        >>> from skcomplex.models.ecology import ExponentialGrowth, Population
        >>> from skcomplex.core.runtime import Simulation
        >>> from skcomplex.core.viz import RerunObserver
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
            rr.spawn(executable_path=_viewer_executable(), recording=self.recording_)
        elif self.connect:
            self.recording_.connect_grpc()
        self._visuals: list[tuple[CompiledElement, Visual]] = [
            (element, resolve_visual(element.visual)) for element in problem.elements
        ]
        self._agent_visuals: dict[str, AgentsVisual] = {}
        for agents in problem.agents:
            visual = resolve_visual(agents.visual)
            self._agent_visuals[agents.path] = visual if isinstance(visual, AgentsVisual) else AgentsVisual()
        self._space = problem.agent_space
        if isinstance(self._space, NetworkSpace):
            edges = [(str(first), str(second)) for first, second in self._space.graph.edges]
            for agents in problem.agents:
                self.recording_.log(agents.path, rr.GraphEdges(edges, graph_type='undirected'), static=True)
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
        for path, snapshot in frame.agents.items():
            visual = self._agent_visuals.setdefault(path, AgentsVisual())
            visual.log_agents(self.recording_, path, snapshot, self._space)
        for name, value in frame.reporters.items():
            self.recording_.log(f'reporters/{name}', rr.Scalars(np.atleast_1d(value)))
        for name, value in frame.observables.items():
            self.recording_.log(name, rr.Scalars(np.atleast_1d(np.abs(value) if np.iscomplexobj(value) else value)))
        for name in frame.problem.parameters.names:
            self.recording_.log(f'parameters/{name}', rr.Scalars(np.atleast_1d(frame.value(name))))
        self.logged_frames_ += 1

    def on_stop(self) -> None:
        """Flush the recording."""
        self.recording_.flush()
