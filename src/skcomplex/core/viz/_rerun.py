"""Stream simulations to the Rerun viewer or to a recording file."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from ..base import ValidationError
from ..compile import CompiledElement, Problem, RenewalRecord
from ..model import NetworkSpace
from ..runtime import Frame, Observer
from ._blueprints import default_blueprint, save_blueprint
from ._traits import agents_visual_of
from ._visuals import AgentsVisual, Visual, _log_scalars, resolve_visual

RENEWAL_MEMBERS = 20
EXTRA = 'extra'
UNSAFE = re.compile(r'[^A-Za-z0-9_/.-]')


def _viewer_executable() -> str | None:
    """Return the Rerun viewer installed next to the running interpreter, or `None` to search `PATH`."""
    for name in ('rerun', 'rerun.exe'):
        candidate = Path(sys.executable).parent / name
        if candidate.is_file():
            return str(candidate)
    return None


def _sanitised(path: str) -> str:
    """Return an entity path with every character Rerun would escape, such as a space, replaced by an underscore."""
    return UNSAFE.sub('_', path)


def _entity_path(name: str, owners: set[str]) -> str:
    """Return the entity path of an observable, under its element when it is an element's dotted output."""
    owner, _, leaf = name.rpartition('.')
    return _sanitised(f'{owner}/{leaf}' if owner in owners else name)


def _renewal_text(record: RenewalRecord) -> str:
    """Return the text log entry of the renewals of one pass, with the members that renewed."""
    shown = ', '.join(str(member) for member in record.members[:RENEWAL_MEMBERS].tolist())
    more = f' and {record.members.size - RENEWAL_MEMBERS} more' if record.members.size > RENEWAL_MEMBERS else ''
    return f'{record.members.size} renewals up to t = {record.time:.6g} of members {shown}{more}'


class RerunObserver(Observer):
    """Observer that logs frames to Rerun, with entity paths that follow the system tree.

    Each logged frame sets the `sim_time` and `step` timelines, logs every element with the visual its type
    declares, logs every parameter under `parameters/<name>`, and logs the renewals of renewal laws since the
    previous frame as text entries under `renewals/<label>`.

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

        blueprint:
            The layout of the viewer: `None` to let the viewer choose it, `'default'` for `default_blueprint` of the
            simulated system, or a Rerun blueprint. It is sent with the recording, and written next to a saved
            recording as a `.rbl` file with the same stem.

    Attributes:
        recording_:
            The Rerun recording stream.

        blueprint_:
            The blueprint sent with the recording, or `None`.

    Raises:
        ValidationError:
            The blueprint is a string other than `'default'`.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from skcomplex.core.model import System
        >>> from skcomplex.models.biology.ecology import ExponentialGrowth, Population
        >>> from skcomplex.core.runtime import Simulation
        >>> from skcomplex.core.viz import RerunObserver
        >>> system = System(components=[Population('a', n=1)], interactions=[ExponentialGrowth('g', target='a', r=1)])
        >>> path = Path(tempfile.mkdtemp()) / 'growth.rrd'
        >>> _ = Simulation(system, dt=0.1, observers=[RerunObserver('growth', save_path=path)]).run(until=1)
        >>> path.exists()
        True
        >>> observer = RerunObserver('growth', save_path=path, blueprint='default')
        >>> _ = Simulation(system, dt=0.1, observers=[observer]).run(until=1)
        >>> path.with_suffix('.rbl').exists()
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
        blueprint: Literal['default'] | rrb.Blueprint | None = None,
    ) -> None:
        self.application_id = application_id
        self.spawn = spawn
        self.connect = connect
        self.save_path = save_path
        self.interval = interval
        self.blueprint = blueprint

    def _resolve_blueprint(self, problem: Problem) -> rrb.Blueprint | None:
        """Return the blueprint the observer sends, building the default one when it is asked for."""
        if self.blueprint is None or isinstance(self.blueprint, rrb.Blueprint):
            return self.blueprint
        if self.blueprint == 'default':
            return default_blueprint(problem)
        error_msg = f'blueprint = {self.blueprint!r} is not supported. Use None, \'default\', or a Rerun blueprint.'
        raise ValidationError(error_msg)

    def on_start(self, problem: Problem) -> None:
        """Create the recording and send it to its sinks.

        Args:
            problem:
                The compiled problem.
        """
        self.blueprint_ = self._resolve_blueprint(problem)
        self.recording_ = rr.RecordingStream(self.application_id)
        if self.save_path is not None:
            self.recording_.save(self.save_path, default_blueprint=self.blueprint_)
            if self.blueprint_ is not None:
                path = Path(self.save_path).with_suffix('.rbl')
                save_blueprint(self.blueprint_, path, application_id=self.application_id)
        if self.spawn:
            rr.spawn(executable_path=_viewer_executable(), recording=self.recording_, default_blueprint=self.blueprint_)
        elif self.connect:
            self.recording_.connect_grpc(default_blueprint=self.blueprint_)
        self._visuals: list[tuple[CompiledElement, Visual]] = [
            (element, resolve_visual(element.visual)) for element in problem.elements
        ]
        self._agent_visuals: dict[str, AgentsVisual] = {}
        for agents in problem.agents:
            self._agent_visuals[agents.path] = agents_visual_of(agents, problem)
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
        owners = {element.path for element in frame.problem.elements}
        for name, value in frame.observables.items():
            values = np.atleast_1d(np.abs(value) if np.iscomplexobj(value) else value)
            self.recording_.log(_entity_path(name, owners), rr.Scalars(values))
        for name in frame.problem.parameters.names:
            _log_scalars(self.recording_, f'parameters/{name}', frame.value(name))
        for record in frame.events:
            if isinstance(record, RenewalRecord):
                self.recording_.log(f'renewals/{record.label}', rr.TextLog(_renewal_text(record)))
        self.logged_frames_ += 1

    def log_extra(self, path: str, archetype: rr.AsComponents, *, static: bool = False) -> None:
        """Log an archetype of the caller's own under `extra/<path>`, apart from the paths of the system's entities.

        Characters that Rerun would escape, such as spaces, become underscores, so a name such as `chain angle` needs
        no escaping and the entity never merges into the view of an element.

        Args:
            path:
                The path below `extra`.

            archetype:
                The Rerun archetype to log.

            static:
                Whether the archetype holds for the whole recording.
        """
        self.recording_.log(f'{EXTRA}/{_sanitised(path)}', archetype, static=static)

    def on_stop(self) -> None:
        """Flush the recording."""
        self.recording_.flush()
