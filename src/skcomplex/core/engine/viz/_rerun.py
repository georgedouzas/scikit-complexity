"""Stream simulations to the Rerun viewer or to a recording file."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
import os
import re
import sys
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun import bindings

from ...components.spaces import NetworkSpace
from ...components.specs import CompiledAgentsSpec, CompiledElementSpec
from ...errors import ValidationError
from ...specs import RenewalRecordSpec
from ..problems import BaseProblemSpec, FrameSpec
from ..problems._learning_problem_spec import LearningProblemSpec
from ..runtime import BaseObserver
from ._agents import AgentsVisual
from ._axes import AxisVisual
from ._base import BaseVisual
from ._bodies import BodiesVisual
from ._cayley import CayleyVisual
from ._cells import CellVisual
from ._expectation import ExpectationVisual
from ._geometry import GeometryVisual
from ._histograms import HistogramVisual
from ._images import ImageVisual
from ._linear_programs import LinearProgramVisual
from ._matrices import MATRIX_RANK, MatrixVisual
from ._meshes import MeshVisual
from ._points import PointsVisual
from ._rings import PhaseRingVisual
from ._scalars import ScalarsVisual
from ._sites import SiteVisual

_RENEWAL_MEMBERS = 20
_VIEWER_FLUSH_SECONDS = 30.0
_FILE_FLUSH_TICK_SECONDS = 10**9
_EXTRA = 'extra'
_UNSAFE = re.compile(r'[^A-Za-z0-9_/.-]')
_MAX_SERIES_PER_VIEW = 6
_MAX_VIEWS_PER_ROW = 3
_SERIES_WIDTH = 2.5
_SPATIAL_ROW_SHARE = 2.0
_SERIES_ROW_SHARE = 1.0
_WHOLE_RUN = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
_VISUALS: dict[str, type[BaseVisual]] = {
    'agents': AgentsVisual,
    'axes': AxisVisual,
    'bodies': BodiesVisual,
    'cayley': CayleyVisual,
    'cell': CellVisual,
    'expectations': ExpectationVisual,
    'geometry': GeometryVisual,
    'histograms': HistogramVisual,
    'image': ImageVisual,
    'linear_program': LinearProgramVisual,
    'matrix': MatrixVisual,
    'mesh': MeshVisual,
    'phase_ring': PhaseRingVisual,
    'points': PointsVisual,
    'scalars': ScalarsVisual,
    'sites': SiteVisual,
}


class RerunObserver(BaseObserver):
    """Observer that logs frames to Rerun.

    Its entity paths follow the system tree. `prepare(problem)` creates the recording, builds the visual of every
    element and agent group from the visual its type declares, lays out their views, and sends the recording to its
    sinks. `run(frame)` logs a frame, and `run(final=True)` flushes the recording. Each logged frame sets the `sim_time`
    and `step` timelines, logs every element with its visual, logs every parameter up to a matrix under
    `parameters/<name>`, leaving out coupling tensors, whose entries no view can draw, and logs the renewals of renewal
    laws since the previous frame as text entries under `renewals/<label>`. A subclass logs entities of its own apart
    from the system's with `_log_extra`.

    The default blueprint puts spatial views first and side by side, with time series views below them. Scalar elements
    of one type share a time series view with at most six series, and time series views show the whole run. The side
    panels are collapsed, the top panel is hidden, and the time panel follows the latest frame of the simulated time.

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
            The layout of the viewer: `None` to let the viewer choose it, `'default'` for the default blueprint built
            from the views of the visuals, or a Rerun blueprint. It is sent with the recording, and written next to a
            saved recording as a `.rbl` file with the same stem.

        primary:
            The name or the origin of the view to put first in the default blueprint, with its row, or `None` for the
            first spatial view, or the first time series view without one.

        parameters:
            Whether the default blueprint has a time series view of the parameters.

    Attributes:
        recording_:
            The Rerun recording stream.

        visuals_:
            The prepared visuals of the elements and the agent groups, in the order they are logged.

        views_:
            The views of the visuals, the observables, and the reporters, in the order of the default blueprint, so the
            first one is the primary view that shows the signature behaviour of the system.

        blueprint_:
            The blueprint sent with the recording, or `None`.

        logged_frames_:
            The number of frames logged.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems.biology.common.populations import ExponentialGrowthInteraction, PopulationElement
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.engine.viz import RerunObserver
        >>> system = System(
        ...     elements=[PopulationElement('a', n=1)],
        ...     interactions=[ExponentialGrowthInteraction('g', target='a', r=1)],
        ... )
        >>> path = Path(tempfile.mkdtemp()) / 'growth.rrd'
        >>> _ = simulate(system, until=1, dt=0.1, observers=[RerunObserver('growth', save_path=path)])
        >>> path.exists()
        True
        >>> observer = RerunObserver('growth', save_path=path, blueprint='default')
        >>> _ = simulate(system, until=1, dt=0.1, observers=[observer])
        >>> path.with_suffix('.rbl').exists()
        True

        The default blueprint of a compiled problem, with a chosen view first:
        >>> from skcomplex.systems.chemistry.pattern_formation import GrayScottSystem
        >>> problem = simulate(GrayScottSystem(shape=(16, 16)), until=0).problem
        >>> observer = RerunObserver(blueprint='default', primary='cells/v').prepare(problem)
        >>> [str(view.origin) for view in observer.blueprint_.root_container.contents]
        ['cells/v', 'cells/u']
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
        primary: str | None = None,
        parameters: bool = False,
    ) -> None:
        self.application_id = application_id
        self.spawn = spawn
        self.connect = connect
        self.save_path = save_path
        self.interval = interval
        self.blueprint = blueprint
        self.primary = primary
        self.parameters = parameters

    def _check_settings(self) -> None:
        """Raise an error for an unknown blueprint or an unused setting.

        A known blueprint is `None`, `'default'`, or a Rerun blueprint.

        Raises:
            ValidationError:
                The blueprint is a string other than `'default'`, the primary view or the parameters view is given
                without the default blueprint, or the interval is not a positive integer.
        """
        if not (self.blueprint is None or self.blueprint == 'default' or isinstance(self.blueprint, rrb.Blueprint)):
            error_msg = f'blueprint = {self.blueprint!r} is not supported. Use None, \'default\', or a Rerun blueprint.'
            raise ValidationError(error_msg)
        if self.blueprint != 'default' and (self.primary is not None or self.parameters):
            error_msg = 'primary and parameters lay out the default blueprint. Pass them with blueprint=\'default\'.'
            raise ValidationError(error_msg)
        if isinstance(self.interval, bool) or not isinstance(self.interval, int) or self.interval < 1:
            error_msg = f'interval = {self.interval!r} must be a positive integer.'
            raise ValidationError(error_msg)

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Create the recording, build the visuals, and send the recording to its sinks.

        It first raises an error for an unknown blueprint or an unused setting. There is one visual for every element
        and agent group.

        Args:
            context:
                The compiled problem the run advances.

        Raises:
            ValidationError:
                A setting is invalid, no problem is passed, or no view has the name or the origin of `primary`.
        """
        self._check_settings()
        if context is None:
            error_msg = 'A Rerun observer prepares on a compiled problem. Pass the problem to `prepare`.'
            raise ValidationError(error_msg)
        problem = context
        # Rerun keeps the threads of a recording until its orphans are cleaned up, so a process running many
        # simulations, such as the docs build, would otherwise run out of threads.
        bindings.flush_and_cleanup_orphaned_recordings()
        self.recording_ = rr.RecordingStream(self.application_id, batcher_config=self._batcher_config())
        agents = problem.agents
        self.visuals_: list[BaseVisual] = [
            *(self._visual_of(element).prepare(problem) for element in problem.elements),
            *(self._agents_visual_of(group, problem).prepare(problem) for group in agents),
            *self._matrix_visuals(problem),
        ]
        spatial, series = self._ordered_views(problem)
        rows = [spatial, series] if self.primary is None else self._with_primary(spatial, series, self.primary)
        self.views_ = [view for row in rows for view in row]
        self.blueprint_ = self._default_blueprint(rows, spatial) if self.blueprint == 'default' else self.blueprint
        if self.save_path is not None:
            self.recording_.save(self.save_path, default_blueprint=self.blueprint_)
            if self.blueprint_ is not None:
                self.blueprint_.save(self.application_id, Path(self.save_path).with_suffix('.rbl'))
        if self.spawn and self._display_is_missing():
            warning_msg = 'The Rerun viewer needs a display, and this machine has none, so no viewer opens.'
            warnings.warn(warning_msg, UserWarning, stacklevel=2)
        elif self.spawn:
            rr.spawn(
                executable_path=self._viewer_executable(),
                recording=self.recording_,
                default_blueprint=self.blueprint_,
            )
        elif self.connect:
            self.recording_.connect_grpc(default_blueprint=self.blueprint_)
        space = problem.agent_space
        if isinstance(space, NetworkSpace):
            edges = [(str(first), str(second)) for first, second in space.graph.edges]
            for group in agents:
                self.recording_.log(group.path, rr.GraphEdges(edges, graph_type='undirected'), static=True)
        self.logged_frames_ = 0

    @staticmethod
    def _display_is_missing() -> bool:
        """Return whether this is a Linux machine without a display.

        There the viewer exits and Rerun waits for it.
        """
        names = ('DISPLAY', 'WAYLAND_DISPLAY', 'WAYLAND_SOCKET')
        return sys.platform == 'linux' and not any(os.environ.get(name) for name in names)

    def _visual_of(self, element: CompiledElementSpec | CompiledAgentsSpec) -> BaseVisual:
        """Return the visual the spec of an element names.

        It is built with the options of the spec.
        """
        spec = element.visual
        visual = _VISUALS[spec.kind] if isinstance(spec.kind, str) else spec.kind
        return visual(**dict(spec.options), path=element.path, recording=self.recording_)

    def _agents_visual_of(self, agents: CompiledAgentsSpec, problem: BaseProblemSpec) -> AgentsVisual:
        """Return the visual of an agent group.

        It is its declared agents visual, or the histogram visual without a space.
        """
        visual = self._visual_of(agents)
        if not isinstance(visual, AgentsVisual):
            visual = AgentsVisual(path=agents.path, recording=self.recording_)
        if problem.agent_space is None and type(visual) is AgentsVisual and visual.position is None:
            return HistogramVisual(
                color_by=visual.color_by,
                color_range=visual.color_range,
                path=agents.path,
                recording=self.recording_,
            )
        return visual

    def _scalar_views(self, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return the time series views of the scalar elements.

        There is one per element type with at most six series each.
        """
        groups: dict[str, list[tuple[CompiledElementSpec, rrb.View]]] = {}
        for visual in self.visuals_:
            if isinstance(visual, ScalarsVisual):
                element = visual.element_
                groups.setdefault(self._type_name(element), []).extend((element, view) for view in visual.views_)
        views: list[rrb.View] = []
        for name, members in groups.items():
            chunks = list(self._chunks(members, problem))
            for number, chunk in enumerate(chunks, start=1):
                label = name if len(chunks) == 1 else f'{name} {number}'
                views.append(self._series_view([view for _, view in chunk], label))
        return views

    def _element_views(self) -> tuple[list[rrb.View], list[rrb.View]]:
        """Return the spatial and time series views of the elements of other visuals.

        Points share one spatial view per dimension, since the elements of a system share its space.
        """
        spatial: list[rrb.View] = []
        series: list[rrb.View] = []
        points: dict[type, list[rrb.View]] = {}
        for visual in self.visuals_:
            if isinstance(visual, ScalarsVisual) or isinstance(visual.element_, CompiledAgentsSpec):
                continue
            for view in visual.views_:
                if isinstance(view, rrb.TimeSeriesView):
                    series.append(view)
                elif isinstance(visual, PointsVisual):
                    points.setdefault(type(view), []).append(view)
                else:
                    spatial.append(view)
        return [*spatial, *(self._merged(views, 'points') for views in points.values())], series

    def _agent_views(self, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return the views of the agents.

        They are merged into one per view class when the groups share a space.
        """
        agents = [visual for visual in self.visuals_ if isinstance(visual.element_, CompiledAgentsSpec)]
        views = [view for visual in agents for view in visual.views_]
        if problem.agent_space is None:
            return views
        by_class: dict[type, list[rrb.View]] = {}
        for view in views:
            by_class.setdefault(type(view), []).append(view)
        return [self._merged(group, 'agents') for group in by_class.values()]

    def _summary_views(self, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return the time series views of the observables and the agent reporters.

        Optionally, the parameters get views too.
        """
        views: list[rrb.View] = []
        paths = [f'{scope}/{item.label}' if scope else item.label for scope, item in problem.observables]
        problems = (problem, *problem.partition_problems)
        paths += [f'{renewal.label}.renewals' for item in problems for renewal in getattr(item, 'renewals', ())]
        if paths:
            views.append(self._series_view([rrb.TimeSeriesView(origin=f'/{path}') for path in paths], 'observables'))
        if problem.agents:
            views.append(self._series_view([rrb.TimeSeriesView(origin='/reporters')], 'reporters'))
        if self.parameters and problem.parameters.names:
            views.append(self._series_view([rrb.TimeSeriesView(origin='/parameters')], 'parameters'))
        return views

    def _matrix_visuals(self, problem: BaseProblemSpec) -> list[BaseVisual]:
        """Return the heat maps of the trained matrices of a system that trains.

        Other problems get none.
        """
        if not isinstance(problem, LearningProblemSpec):
            return []
        matrices = [entry.name for entry in problem.states.entries if len(entry.shape) == MATRIX_RANK]
        return [
            MatrixVisual(name.rpartition('.')[2], path=name.rpartition('.')[0], recording=self.recording_).prepare(
                problem,
            )
            for name in matrices
        ]

    def _learning_views(self, problem: LearningProblemSpec) -> list[rrb.View]:
        """Return the time series view of the loss and the metrics over epochs.

        The heat maps of the matrices follow it.
        """
        loss = rrb.TimeSeriesView(origin=f'/reporters/{problem.training}', name='loss')
        return [loss, *(view for visual in self.visuals_ for view in visual.views_)]

    def _ordered_views(self, problem: BaseProblemSpec) -> tuple[list[rrb.View], list[rrb.View]]:
        """Return the spatial views and the time series views of a problem.

        Each comes in order of importance. A system that trains has no spatial view of its own, and its loss leads the
        heat maps of its matrices.
        """
        if isinstance(problem, LearningProblemSpec):
            return [], [self._whole_run(view) for view in self._learning_views(problem)]
        spatial, series = self._element_views()
        spatial = [*spatial, *self._agent_views(problem)]
        series = [*self._scalar_views(problem), *series, *self._summary_views(problem)]
        return [self._whole_run(view) for view in spatial], [self._whole_run(view) for view in series]

    def _default_blueprint(self, rows: list[list[rrb.View]], spatial: list[rrb.View]) -> rrb.Blueprint:
        """Return the blueprint of rows of views.

        A row of spatial views is taller than a row of time series views.
        """
        containers = [self._row(views) for views in rows if views]
        if len(containers) > 1:
            shares = [_SPATIAL_ROW_SHARE, _SERIES_ROW_SHARE] if rows[0] is spatial else [_SERIES_ROW_SHARE] * 2
            root: rrb.View | rrb.Container = rrb.Vertical(*containers, row_shares=shares)
        else:
            root = containers[0] if containers else rrb.Vertical()
        return rrb.Blueprint(
            root,
            rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
            rrb.TopPanel(state='hidden'),
            collapse_panels=True,
        )

    def _run(self, frame: FrameSpec | None = None) -> None:
        """Log a frame when its step is a multiple of the interval.

        It logs nothing for the call that ends the run.
        """
        if frame is None or frame.step % self.interval:
            return
        self.recording_.set_time('step', sequence=frame.step)
        self.recording_.set_time('sim_time', duration=frame.time)
        for visual in self.visuals_:
            visual.run(frame)
        for name, value in frame.reporters.items():
            self.recording_.log(f'reporters/{name}', rr.Scalars(np.atleast_1d(value)))
        owners = {element.path for element in frame.problem.elements}
        for name, observed in frame.observables.items():
            values = np.atleast_1d(np.abs(observed) if np.iscomplexobj(observed) else observed)
            self.recording_.log(self._entity_path(name, owners), rr.Scalars(values))
        for name in frame.problem.parameters.names:
            parameter = frame.value(name)
            if np.ndim(parameter) <= MATRIX_RANK:
                BaseVisual._log_scalars(self.recording_, f'parameters/{name}', parameter)
        for record in frame.events:
            if isinstance(record, RenewalRecordSpec):
                self.recording_.log(f'renewals/{record.label}', rr.TextLog(self._renewal_text(record)))
        self.logged_frames_ += 1

    def _batcher_config(self) -> rr.ChunkBatcherConfig:
        """Return the batching of the recording.

        It batches by size alone when no viewer waits for the frames. Rerun flushes its batches every 200 ms by default,
        which writes a chunk per entity at each flush, so on a slower machine a recording of many entities splits into
        more chunks and its file grows several times. A recording that only goes to a file flushes when its batch is
        large enough, so its size does not depend on the machine.
        """
        config = rr.ChunkBatcherConfig.DEFAULT()
        if not (self.spawn or self.connect):
            config.flush_tick = _FILE_FLUSH_TICK_SECONDS
        return config

    def _release(self) -> None:
        """Flush the recording, and warn after a while about an unreachable viewer.

        It does not wait forever for a viewer that cannot be reached.
        """
        if not (self.spawn or self.connect):
            self.recording_.flush()
            return
        try:
            self.recording_.flush(timeout_sec=_VIEWER_FLUSH_SECONDS)
        except RuntimeError as error:
            warning_msg = f'The Rerun viewer did not receive the whole recording: {error}'
            warnings.warn(warning_msg, UserWarning, stacklevel=2)

    def _log_extra(self, path: str, archetype: rr.AsComponents, *, static: bool = False) -> None:
        """Log an archetype of a subclass's own under `extra/<path>`.

        It stays apart from the paths of the system's entities. Characters that Rerun would escape, such as spaces,
        become underscores, so a name such as `chain angle` needs no escaping and the entity never merges into the view
        of an element.

        Args:
            path:
                The path below `extra`.

            archetype:
                The Rerun archetype to log.

            static:
                Whether the archetype holds for the whole recording.
        """
        self.recording_.log(f'{_EXTRA}/{self._sanitised(path)}', archetype, static=static)

    @staticmethod
    def _sanitised(path: str) -> str:
        """Return an entity path with every character Rerun would escape replaced.

        Such a character, like a space, is replaced by an underscore.
        """
        return _UNSAFE.sub('_', path)

    @staticmethod
    def _entity_path(name: str, owners: set[str]) -> str:
        """Return the entity path of an observable.

        It is under its element when it is an element's dotted output.
        """
        owner, _, leaf = name.rpartition('.')
        return RerunObserver._sanitised(f'{owner}/{leaf}' if owner in owners else name)

    @staticmethod
    def _renewal_text(record: RenewalRecordSpec) -> str:
        """Return the text log entry of the renewals of one pass.

        It names the members that renewed.
        """
        shown = ', '.join(str(member) for member in record.members[:_RENEWAL_MEMBERS].tolist())
        more = f' and {record.members.size - _RENEWAL_MEMBERS} more' if record.members.size > _RENEWAL_MEMBERS else ''
        return f'{record.members.size} renewals up to t = {record.time:.6g} of members {shown}{more}'

    @staticmethod
    def _viewer_executable() -> str | None:
        """Return the Rerun viewer installed next to the running interpreter.

        It returns `None` to search `PATH`.
        """
        for name in ('rerun', 'rerun.exe'):
            candidate = Path(sys.executable).parent / name
            if candidate.is_file():
                return str(candidate)
        return None

    @staticmethod
    def _contents(view: rrb.View) -> list[str]:
        """Return the entity path expressions of a view with its origin written out."""
        contents = view.contents if isinstance(view.contents, list | tuple) else [view.contents]
        origin = '/' + str(view.origin).strip('/')
        expressions = [str(expression).replace('$origin', origin).replace('//', '/') for expression in contents]
        return [expression if expression[0] in '+-/' else f'/{expression}' for expression in expressions]

    @staticmethod
    def _merged(views: list[rrb.View], name: str) -> rrb.View:
        """Return one view that shows the contents of every view.

        It is of the class of the first view.
        """
        if len(views) == 1:
            return views[0]
        contents = [expression for view in views for expression in RerunObserver._contents(view)]
        kind: Callable[..., rrb.View] = type(views[0])
        return kind(origin='/', contents=contents, name=name)

    @staticmethod
    def _series_view(views: list[rrb.View], name: str) -> rrb.TimeSeriesView:
        """Return a time series view of several time series views over the whole run.

        Its lines are wide enough to read.
        """
        contents = [expression for view in views for expression in RerunObserver._contents(view)]
        return rrb.TimeSeriesView(
            origin='/',
            contents=contents,
            name=name,
            defaults=[rr.SeriesLines.from_fields(widths=_SERIES_WIDTH)],
            axis_x=_WHOLE_RUN,
        )

    @staticmethod
    def _whole_run(view: rrb.View) -> rrb.View:
        """Return a time series view with the whole-run time axis and readable lines.

        Any other view is returned unchanged.
        """
        if not isinstance(view, rrb.TimeSeriesView) or 'TimeAxis' in view.properties:
            return view
        legend = view.properties.get('PlotLegend')
        return rrb.TimeSeriesView(
            origin=view.origin,
            contents=view.contents,
            name=view.name,
            defaults=[rr.SeriesLines.from_fields(widths=_SERIES_WIDTH)],
            axis_x=_WHOLE_RUN,
            plot_legend=legend if isinstance(legend, rrb.PlotLegend) else None,
        )

    @staticmethod
    def _type_name(element: CompiledElementSpec) -> str:
        """Return the name of the element type of an element or of the members of a group."""
        kind = getattr(element.element, 'element', None) or type(element.element)
        return str(getattr(kind, '__name__', type(element.element).__name__))

    @staticmethod
    def _chunks(
        elements: list[tuple[CompiledElementSpec, rrb.View]],
        problem: BaseProblemSpec,
    ) -> Iterator[list[tuple[CompiledElementSpec, rrb.View]]]:
        """Yield consecutive runs of elements whose series fit in one view.

        Lag states are left out.
        """
        chunk: list[tuple[CompiledElementSpec, rrb.View]] = []
        count = 0
        for element, view in elements:
            shown = [name for name in element.states.values() if problem.companions.get(name, ('',))[0] != 'lag']
            series = len(shown) * (element.members or 1)
            if chunk and count + series > _MAX_SERIES_PER_VIEW:
                yield chunk
                chunk, count = [], 0
            chunk.append((element, view))
            count += series
        if chunk:
            yield chunk

    @staticmethod
    def _with_primary(spatial: list[rrb.View], series: list[rrb.View], primary: str) -> list[list[rrb.View]]:
        """Return the rows of views with the view of a name or an origin first.

        It comes first in the first row.
        """
        for row, other in ((spatial, series), (series, spatial)):
            found = [view for view in row if primary.strip('/') in {str(view.name), str(view.origin).strip('/')}]
            if found:
                return [[found[0], *[view for view in row if view is not found[0]]], other]
        names = ', '.join(str(view.name) for view in [*spatial, *series])
        error_msg = f'No view is named or has the origin {primary!r}. The views are: {names}.'
        raise ValidationError(error_msg)

    @staticmethod
    def _row(views: list[rrb.View]) -> rrb.View | rrb.Container:
        """Return a single view, views side by side, or a grid of views."""
        if len(views) == 1:
            return views[0]
        if len(views) <= _MAX_VIEWS_PER_ROW:
            return rrb.Horizontal(*views)
        return rrb.Grid(*views, grid_columns=math.ceil(math.sqrt(len(views))))
