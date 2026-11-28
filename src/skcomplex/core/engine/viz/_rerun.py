"""Stream simulations to the Rerun viewer or to a recording file."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
import os
import re
import sys
import warnings
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun import bindings
from sage.all__sagemath_symbolics import SR

from ...components.elements import VisualSpec
from ...components.spaces import NetworkSpace
from ...errors import ValidationError
from ..problems import BaseProblemSpec, CompiledElementSpec, FrameSpec
from ..problems._base_problem_spec import _CompiledAgents
from ..problems._declared_element import _DeclaredElement
from ..problems._group_element import _GroupElement
from ..problems._learning_problem_spec import LearningProblemSpec
from ..problems._renewal_kernel import _RenewalRecordSpec
from ..runtime import BaseObserver
from ._agents import AgentsVisual
from ._axes import AxisVisual
from ._base import BaseVisual
from ._bodies import BodiesVisual
from ._cells import CellVisual
from ._expectation import ExpectationVisual
from ._histograms import HistogramVisual
from ._images import ImageVisual
from ._linear_programs import LinearProgramVisual
from ._matrices import MATRIX_RANK, MatrixVisual
from ._meshes import MeshVisual
from ._points import PointsVisual
from ._rings import PhaseRingVisual
from ._scalars import ScalarsVisual
from ._sites import SiteVisual

RENEWAL_MEMBERS = 20
VIEWER_FLUSH_SECONDS = 30.0
FILE_FLUSH_TICK_SECONDS = 10**9
EXTRA = 'extra'
UNSAFE = re.compile(r'[^A-Za-z0-9_/.-]')
MAX_SERIES_PER_VIEW = 6
MAX_VIEWS_PER_ROW = 3
SERIES_WIDTH = 2.5
SPATIAL_ROW_SHARE = 2.0
SERIES_ROW_SHARE = 1.0
WHOLE_RUN = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
VISUALS: dict[str, type[BaseVisual]] = {
    'agents': AgentsVisual,
    'axes': AxisVisual,
    'bodies': BodiesVisual,
    'cell': CellVisual,
    'expectations': ExpectationVisual,
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
    """An observer that logs the frames of a run to the Rerun viewer.

    It draws every element of a system with the visual its class names, in the Rerun viewer or in a `.rrd` file, such
    as `system.simulate(until=10.0, observers=[RerunObserver(save_path='run.rrd')])`. `prepare(problem)` builds the
    visuals and lays out their views, `run(frame)` draws one frame, and `run(final=True)` writes out what is left. Each
    frame also draws the parameters and the renewals of renewal laws, on timelines of the simulated time and the step.
    The default layout puts the pictures of space first, side by side, and the lines over time below them. A subclass
    draws more with `_log_extra`. It replays a predator and prey cycle, the spread of an epidemic on a network, and the
    patterns of a chemical reaction.

    Read more in the user guide, under Observe.

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

        visuals:
            How to draw elements by path, such as `{'oscillators': VisualSpec('phase_ring', {'phase': 'theta'})}`, in
            place of the drawing each element has by default, or `None` to keep every default.

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

    See Also:
        BaseVisual: Draws one element at every frame.

    Examples:
        >>> from skcomplex.core.engine.viz import RerunObserver
        >>> from skcomplex.systems.biology.ecology import LotkaVolterraSystem
        >>> observer = RerunObserver(interval=10)
        >>> _ = LotkaVolterraSystem().prepare().simulate(until=1.0, dt=0.1, observers=[observer])
        >>> observer.logged_frames_
        2
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
        visuals: Mapping[str, VisualSpec] | None = None,
    ) -> None:
        self.application_id = application_id
        self.spawn = spawn
        self.connect = connect
        self.save_path = save_path
        self.interval = interval
        self.blueprint = blueprint
        self.primary = primary
        self.parameters = parameters
        self.visuals = visuals

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

    def _check_visuals(self, problem: BaseProblemSpec) -> None:
        """Raise an error for a drawing whose path names no element of the problem."""
        paths = [element.path for element in problem.elements] + [group.path for group in problem.agents]
        unknown = sorted(set(self.visuals or {}) - set(paths))
        if unknown:
            error_msg = (
                f'visuals names {", ".join(map(repr, unknown))}, which is not an element of the system. '
                f'The elements are {", ".join(map(repr, paths))}.'
            )
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
        self._check_visuals(problem)
        # Rerun keeps the threads of a recording until its orphans are cleaned up, so a process running many
        # simulations, such as the docs build, would otherwise run out of threads.
        bindings.flush_and_cleanup_orphaned_recordings()
        self.recording_ = rr.RecordingStream(self.application_id, batcher_config=self._get_batcher_config())
        agents = problem.agents
        self.visuals_: list[BaseVisual] = [
            *(self._get_visual_of(element, problem).prepare(problem) for element in problem.elements),
            *(self._get_agents_visual_of(group, problem).prepare(problem) for group in agents),
            *self._get_matrix_visuals(problem),
        ]
        spatial, series = self._get_ordered_views(problem)
        rows = [spatial, series] if self.primary is None else self._copy_with_primary(spatial, series, self.primary)
        self.views_ = [view for row in rows for view in row]
        self.blueprint_ = self._get_default_blueprint(rows, spatial) if self.blueprint == 'default' else self.blueprint
        if self.save_path is not None:
            self.recording_.save(self.save_path, default_blueprint=self.blueprint_)
            if self.blueprint_ is not None:
                self.blueprint_.save(self.application_id, Path(self.save_path).with_suffix('.rbl'))
        if self.spawn and self._display_is_missing():
            warning_msg = 'The Rerun viewer needs a display, and this machine has none, so no viewer opens.'
            warnings.warn(warning_msg, UserWarning, stacklevel=2)
        elif self.spawn:
            rr.spawn(
                executable_path=self._get_viewer_executable(),
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

    def _get_visual_of(
        self,
        element: CompiledElementSpec | _CompiledAgents,
        problem: BaseProblemSpec,
    ) -> BaseVisual:
        """Return the visual the spec of an element names.

        It is built with the options of the spec. An option the visual lists in `numeric_options`, given as an
        expression of the parameters of the element, such as `'R_0 / 2'`, takes its value at the start of the run.
        """
        spec = (self.visuals or {}).get(element.path, element.visual)
        visual: type[BaseVisual] = VISUALS[spec.kind] if isinstance(spec.kind, str) else spec.kind
        options = {
            name: self._get_option_value(value, element, problem) if name in visual.numeric_options else value
            for name, value in spec.options.items()
        }
        return visual(**options, path=element.path, recording=self.recording_)

    @staticmethod
    def _get_option_value(
        value: object,
        element: CompiledElementSpec | _CompiledAgents,
        problem: BaseProblemSpec,
    ) -> object:
        """Return an option, with an expression of parameters replaced by its value."""
        names: Mapping[str, str] = getattr(element, 'parameters', {})
        if not isinstance(value, str) or not names:
            return value
        try:
            expression = SR(value)
        except (TypeError, ValueError, SyntaxError):
            return value
        symbols = [str(symbol) for symbol in expression.variables()]
        if not symbols or not all(symbol in names for symbol in symbols):
            return value
        values = {
            SR.var(symbol): float(np.ravel(problem.parameters.read(problem.initial_parameters, names[symbol]))[0])
            for symbol in symbols
        }
        return float(expression.subs(values))

    def _get_agents_visual_of(self, agents: _CompiledAgents, problem: BaseProblemSpec) -> AgentsVisual:
        """Return the visual of an agent group.

        It is its declared agents visual, or the histogram visual without a space.
        """
        visual = self._get_visual_of(agents, problem)
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

    def _get_scalar_views(self, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return the time series views of the scalar elements.

        There is one per element type with at most six series each.
        """
        groups: dict[str, list[tuple[CompiledElementSpec, rrb.View]]] = {}
        for visual in self.visuals_:
            if isinstance(visual, ScalarsVisual) and isinstance(visual.element_, CompiledElementSpec):
                element = visual.element_
                groups.setdefault(self._get_type_name(element), []).extend((element, view) for view in visual.views_)
        views: list[rrb.View] = []
        for name, members in groups.items():
            chunks = list(self._get_chunks(members, problem))
            for number, chunk in enumerate(chunks, start=1):
                label = name if len(chunks) == 1 else f'{name} {number}'
                views.append(self._get_series_view([view for _, view in chunk], label))
        return views

    def _get_element_views(self) -> tuple[list[rrb.View], list[rrb.View]]:
        """Return the spatial and time series views of the elements of other visuals.

        Points share one spatial view per dimension, since the elements of a system share its space.
        """
        spatial: list[rrb.View] = []
        series: list[rrb.View] = []
        points: dict[type, list[rrb.View]] = {}
        for visual in self.visuals_:
            if isinstance(visual, ScalarsVisual) or isinstance(visual.element_, _CompiledAgents):
                continue
            for view in visual.views_:
                if isinstance(view, rrb.TimeSeriesView):
                    series.append(view)
                elif isinstance(visual, PointsVisual):
                    points.setdefault(type(view), []).append(view)
                else:
                    spatial.append(view)
        return [*spatial, *(self._get_merged(views, 'points') for views in points.values())], series

    def _get_agent_views(self, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return the views of the agents.

        They are merged into one per view class when the groups share a space.
        """
        agents = [visual for visual in self.visuals_ if isinstance(visual.element_, _CompiledAgents)]
        views = [view for visual in agents for view in visual.views_]
        if problem.agent_space is None:
            return views
        by_class: dict[type, list[rrb.View]] = {}
        for view in views:
            by_class.setdefault(type(view), []).append(view)
        return [self._get_merged(group, 'agents') for group in by_class.values()]

    def _get_summary_views(self, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return the time series views of the observables and the agent reporters.

        Optionally, the parameters get views too.
        """
        views: list[rrb.View] = []
        paths = [f'{scope}/{item.label}' if scope else item.label for scope, item in problem.observables]
        problems = (problem, *problem.partition_problems)
        paths += [f'{renewal.label}.renewals' for item in problems for renewal in getattr(item, 'renewals', ())]
        if paths:
            views.append(
                self._get_series_view([rrb.TimeSeriesView(origin=f'/{path}') for path in paths], 'observables'),
            )
        if problem.agents:
            views.append(self._get_series_view([rrb.TimeSeriesView(origin='/reporters')], 'reporters'))
        if self.parameters and problem.parameters.names:
            views.append(self._get_series_view([rrb.TimeSeriesView(origin='/parameters')], 'parameters'))
        return views

    def _get_matrix_visuals(self, problem: BaseProblemSpec) -> list[BaseVisual]:
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

    def _get_learning_views(self, problem: LearningProblemSpec) -> list[rrb.View]:
        """Return the time series view of the loss and the metrics over epochs.

        The heat maps of the matrices follow it.
        """
        loss = rrb.TimeSeriesView(origin=f'/reporters/{problem.training}', name='loss')
        return [loss, *(view for visual in self.visuals_ for view in visual.views_)]

    def _get_ordered_views(self, problem: BaseProblemSpec) -> tuple[list[rrb.View], list[rrb.View]]:
        """Return the spatial views and the time series views of a problem.

        Each comes in order of importance. A system that trains has no spatial view of its own, and its loss leads the
        heat maps of its matrices.
        """
        if isinstance(problem, LearningProblemSpec):
            return [], [self._get_whole_run(view) for view in self._get_learning_views(problem)]
        spatial, series = self._get_element_views()
        spatial = [*spatial, *self._get_agent_views(problem)]
        series = [*self._get_scalar_views(problem), *series, *self._get_summary_views(problem)]
        return [self._get_whole_run(view) for view in spatial], [self._get_whole_run(view) for view in series]

    def _get_default_blueprint(self, rows: list[list[rrb.View]], spatial: list[rrb.View]) -> rrb.Blueprint:
        """Return the blueprint of rows of views.

        A row of spatial views is taller than a row of time series views.
        """
        containers = [self._get_row(views) for views in rows if views]
        if len(containers) > 1:
            shares = [SPATIAL_ROW_SHARE, SERIES_ROW_SHARE] if rows[0] is spatial else [SERIES_ROW_SHARE] * 2
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
            self.recording_.log(self._get_entity_path(name, owners), rr.Scalars(values))
        for name in frame.problem.parameters.names:
            parameter = frame.get_value(name)
            if np.ndim(parameter) <= MATRIX_RANK:
                BaseVisual._log_scalars(self.recording_, f'parameters/{name}', parameter)
        for record in frame.events:
            if isinstance(record, _RenewalRecordSpec):
                self.recording_.log(f'renewals/{record.label}', rr.TextLog(self._get_renewal_text(record)))
        self.logged_frames_ += 1

    def _get_batcher_config(self) -> rr.ChunkBatcherConfig:
        """Return the batching of the recording.

        It batches by size alone when no viewer waits for the frames. Rerun flushes its batches every 200 ms by default,
        which writes a chunk per entity at each flush, so on a slower machine a recording of many entities splits into
        more chunks and its file grows several times. A recording that only goes to a file flushes when its batch is
        large enough, so its size does not depend on the machine.
        """
        config = rr.ChunkBatcherConfig.DEFAULT()
        if not (self.spawn or self.connect):
            config.flush_tick = FILE_FLUSH_TICK_SECONDS
        return config

    def _release(self) -> None:
        """Flush the recording, and warn after a while about an unreachable viewer.

        It does not wait forever for a viewer that cannot be reached.
        """
        if not (self.spawn or self.connect):
            self.recording_.flush()
            return
        try:
            self.recording_.flush(timeout_sec=VIEWER_FLUSH_SECONDS)
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
        self.recording_.log(f'{EXTRA}/{self._get_sanitised(path)}', archetype, static=static)

    @staticmethod
    def _get_sanitised(path: str) -> str:
        """Return an entity path with every character Rerun would escape replaced.

        Such a character, like a space, is replaced by an underscore.
        """
        return UNSAFE.sub('_', path)

    @staticmethod
    def _get_entity_path(name: str, owners: set[str]) -> str:
        """Return the entity path of an observable.

        It is under its element when it is an element's dotted output.
        """
        owner, _, leaf = name.rpartition('.')
        return RerunObserver._get_sanitised(f'{owner}/{leaf}' if owner in owners else name)

    @staticmethod
    def _get_renewal_text(record: _RenewalRecordSpec) -> str:
        """Return the text log entry of the renewals of one pass.

        It names the members that renewed.
        """
        shown = ', '.join(str(member) for member in record.members[:RENEWAL_MEMBERS].tolist())
        more = f' and {record.members.size - RENEWAL_MEMBERS} more' if record.members.size > RENEWAL_MEMBERS else ''
        return f'{record.members.size} renewals up to t = {record.time:.6g} of members {shown}{more}'

    @staticmethod
    def _get_viewer_executable() -> str | None:
        """Return the Rerun viewer installed next to the running interpreter.

        It returns `None` to search `PATH`.
        """
        for name in ('rerun', 'rerun.exe'):
            candidate = Path(sys.executable).parent / name
            if candidate.is_file():
                return str(candidate)
        return None

    @staticmethod
    def _get_contents(view: rrb.View) -> list[str]:
        """Return the entity path expressions of a view with its origin written out."""
        contents = view.contents if isinstance(view.contents, list | tuple) else [view.contents]
        origin = '/' + str(view.origin).strip('/')
        expressions = [str(expression).replace('$origin', origin).replace('//', '/') for expression in contents]
        return [expression if expression[0] in '+-/' else f'/{expression}' for expression in expressions]

    @staticmethod
    def _get_merged(views: list[rrb.View], name: str) -> rrb.View:
        """Return one view that shows the contents of every view.

        It is of the class of the first view.
        """
        if len(views) == 1:
            return views[0]
        contents = [expression for view in views for expression in RerunObserver._get_contents(view)]
        kind: Callable[..., rrb.View] = type(views[0])
        return kind(origin='/', contents=contents, name=name)

    @staticmethod
    def _get_series_view(views: list[rrb.View], name: str) -> rrb.TimeSeriesView:
        """Return a time series view of several time series views over the whole run.

        Its lines are wide enough to read.
        """
        contents = [expression for view in views for expression in RerunObserver._get_contents(view)]
        return rrb.TimeSeriesView(
            origin='/',
            contents=contents,
            name=name,
            defaults=[rr.SeriesLines.from_fields(widths=SERIES_WIDTH)],
            axis_x=WHOLE_RUN,
        )

    @staticmethod
    def _get_whole_run(view: rrb.View) -> rrb.View:
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
            defaults=[rr.SeriesLines.from_fields(widths=SERIES_WIDTH)],
            axis_x=WHOLE_RUN,
            plot_legend=legend if isinstance(legend, rrb.PlotLegend) else None,
        )

    @staticmethod
    def _get_type_name(element: CompiledElementSpec) -> str:
        """Return the name of the element type of an element or of the members of a group.

        An element declared from a plain element is named after the class of that element, such as `PopulationElement`.
        """
        component = element.element
        member = component._get_member() if isinstance(component, _GroupElement) else component
        origin = member.origin if isinstance(member, _DeclaredElement) else None
        return (origin or type(member)).__name__

    @staticmethod
    def _get_chunks(
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
            if chunk and count + series > MAX_SERIES_PER_VIEW:
                yield chunk
                chunk, count = [], 0
            chunk.append((element, view))
            count += series
        if chunk:
            yield chunk

    @staticmethod
    def _copy_with_primary(spatial: list[rrb.View], series: list[rrb.View], primary: str) -> list[list[rrb.View]]:
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
    def _get_row(views: list[rrb.View]) -> rrb.View | rrb.Container:
        """Return a single view, views side by side, or a grid of views."""
        if len(views) == 1:
            return views[0]
        if len(views) <= MAX_VIEWS_PER_ROW:
            return rrb.Horizontal(*views)
        return rrb.Grid(*views, grid_columns=math.ceil(math.sqrt(len(views))))
