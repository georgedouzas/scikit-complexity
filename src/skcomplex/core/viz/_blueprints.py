"""Lay out the Rerun views of a system as a blueprint."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

from ..base import ValidationError
from ..compile import CompiledAgents, CompiledElement, Problem, compile_system, observables_of
from ..model import Observable, System
from ._traits import agents_visual_of
from ._visuals import PointsVisual, ScalarsVisual, resolve_visual

MAX_SERIES_PER_VIEW = 6
MAX_VIEWS_PER_ROW = 3
SERIES_WIDTH = 2.5
SPATIAL_ROW_SHARE = 2.0
SERIES_ROW_SHARE = 1.0
WHOLE_RUN = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)


def _problem_of(system: System | Problem, observables: Sequence[Observable] = ()) -> Problem:
    """Return the problem of a system with its observables, compiling it when it is not compiled yet."""
    return system if isinstance(system, Problem) else compile_system(system, observables=observables)


def _contents(view: rrb.View) -> list[str]:
    """Return the entity path expressions of a view with its origin written out."""
    contents = view.contents if isinstance(view.contents, list | tuple) else [view.contents]
    origin = '/' + str(view.origin).strip('/')
    expressions = [str(expression).replace('$origin', origin).replace('//', '/') for expression in contents]
    return [expression if expression[0] in '+-/' else f'/{expression}' for expression in expressions]


def _merged(views: list[rrb.View], name: str) -> rrb.View:
    """Return one view of the class of the first view that shows the contents of every view."""
    if len(views) == 1:
        return views[0]
    contents = [expression for view in views for expression in _contents(view)]
    kind: Callable[..., rrb.View] = type(views[0])
    return kind(origin='/', contents=contents, name=name)


def _series_view(views: list[rrb.View], name: str) -> rrb.TimeSeriesView:
    """Return a time series view of several time series views over the whole run, with lines thick enough to read."""
    contents = [expression for view in views for expression in _contents(view)]
    return rrb.TimeSeriesView(
        origin='/',
        contents=contents,
        name=name,
        defaults=[rr.SeriesLines.from_fields(widths=SERIES_WIDTH)],
        axis_x=WHOLE_RUN,
    )


def _whole_run(view: rrb.View) -> rrb.View:
    """Return a time series view with the whole-run time axis and readable lines, and any other view unchanged."""
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


def _type_name(element: CompiledElement) -> str:
    """Return the name of the element type of an element or of the members of a group."""
    kind = getattr(element.element, 'element', None) or type(element.element)
    return str(getattr(kind, '__name__', type(element.element).__name__))


def _series_count(element: CompiledElement, problem: Problem) -> int:
    """Return the number of series a scalars visual logs for an element, leaving out lag states."""
    shown = [name for name in element.states.values() if problem.companions.get(name, ('',))[0] != 'lag']
    return len(shown) * (element.members or 1)


def _chunks(
    elements: list[tuple[CompiledElement, rrb.View]],
    problem: Problem,
) -> Iterator[list[tuple[CompiledElement, rrb.View]]]:
    """Yield consecutive runs of elements whose series fit in one view."""
    chunk: list[tuple[CompiledElement, rrb.View]] = []
    count = 0
    for element, view in elements:
        series = _series_count(element, problem)
        if chunk and count + series > MAX_SERIES_PER_VIEW:
            yield chunk
            chunk, count = [], 0
        chunk.append((element, view))
        count += series
    if chunk:
        yield chunk


def _scalar_views(problem: Problem) -> list[rrb.View]:
    """Return the time series views of the scalar elements, one per element type and at most six series each."""
    groups: dict[str, list[tuple[CompiledElement, rrb.View]]] = {}
    for element in problem.elements:
        visual = resolve_visual(element.visual)
        if not isinstance(visual, ScalarsVisual):
            continue
        for view in visual.views(element, problem):
            groups.setdefault(_type_name(element), []).append((element, view))
    views: list[rrb.View] = []
    for name, members in groups.items():
        chunks = list(_chunks(members, problem))
        for number, chunk in enumerate(chunks, start=1):
            label = name if len(chunks) == 1 else f'{name} {number}'
            views.append(_series_view([view for _, view in chunk], label))
    return views


def _element_views(problem: Problem) -> tuple[list[rrb.View], list[rrb.View]]:
    """Return the spatial views and the time series views of the elements drawn by other visuals.

    Points share one spatial view per dimension, since the elements of a system share its space.
    """
    spatial: list[rrb.View] = []
    series: list[rrb.View] = []
    points: dict[type, list[rrb.View]] = {}
    for element in problem.elements:
        visual = resolve_visual(element.visual)
        if isinstance(visual, ScalarsVisual):
            continue
        for view in visual.views(element, problem):
            if isinstance(view, rrb.TimeSeriesView):
                series.append(view)
            elif isinstance(visual, PointsVisual):
                points.setdefault(type(view), []).append(view)
            else:
                spatial.append(view)
    return [*spatial, *(_merged(views, 'points') for views in points.values())], series


def _agent_views(problem: Problem) -> list[rrb.View]:
    """Return the views of the agents, merged into one per view class when the groups share a space."""
    views: list[rrb.View] = []
    agents: CompiledAgents
    for agents in problem.agents:
        views += agents_visual_of(agents, problem).views(agents, problem)
    if problem.agent_space is None:
        return views
    by_class: dict[type, list[rrb.View]] = {}
    for view in views:
        by_class.setdefault(type(view), []).append(view)
    return [_merged(group, 'agents') for group in by_class.values()]


def _summary_views(problem: Problem, *, parameters: bool) -> list[rrb.View]:
    """Return the time series views of the observables, the agent reporters, and optionally the parameters."""
    views: list[rrb.View] = []
    system = problem.system
    paths = (
        [f'{scope}/{item.label}' if scope else item.label for scope, item in observables_of(system)] if system else []
    )
    problems = (problem, *(partition.problem for partition in problem.partitions))
    paths += [f'{renewal.label}.renewals' for item in problems for renewal in item.renewals]
    if paths:
        views.append(_series_view([rrb.TimeSeriesView(origin=f'/{path}') for path in paths], 'observables'))
    if problem.agents:
        views.append(_series_view([rrb.TimeSeriesView(origin='/reporters')], 'reporters'))
    if parameters and problem.parameters.names:
        views.append(_series_view([rrb.TimeSeriesView(origin='/parameters')], 'parameters'))
    return views


def _ordered_views(problem: Problem, *, parameters: bool) -> tuple[list[rrb.View], list[rrb.View]]:
    """Return the spatial views and the time series views of a problem, each in order of importance."""
    spatial, series = _element_views(problem)
    spatial = [*spatial, *_agent_views(problem)]
    series = [*_scalar_views(problem), *series, *_summary_views(problem, parameters=parameters)]
    return [_whole_run(view) for view in spatial], [_whole_run(view) for view in series]


def _matches(view: rrb.View, primary: str) -> bool:
    """Return whether a view has a name or an origin."""
    return primary.strip('/') in {str(view.name), str(view.origin).strip('/')}


def _with_primary(spatial: list[rrb.View], series: list[rrb.View], primary: str) -> list[list[rrb.View]]:
    """Return the rows of views with the view of a name or an origin first in the first row."""
    for row, other in ((spatial, series), (series, spatial)):
        found = [view for view in row if _matches(view, primary)]
        if found:
            return [[found[0], *[view for view in row if view is not found[0]]], other]
    names = ', '.join(str(view.name) for view in [*spatial, *series])
    error_msg = f'No view is named or has the origin {primary!r}. The views are: {names}.'
    raise ValidationError(error_msg)


def _row(views: list[rrb.View]) -> rrb.View | rrb.Container:
    """Return a single view, views side by side, or a grid of views."""
    if len(views) == 1:
        return views[0]
    if len(views) <= MAX_VIEWS_PER_ROW:
        return rrb.Horizontal(*views)
    return rrb.Grid(*views, grid_columns=math.ceil(math.sqrt(len(views))))


def primary_view(system: System | Problem) -> rrb.View:
    """Return the view that shows the signature behaviour of a system, the first view of its default blueprint.

    Spatial views come first, in the order of the elements and then of the agents, and time series views after them.

    Args:
        system:
            The system, or a problem compiled from one.

    Returns:
        view:
            The primary view.

    Raises:
        ValidationError:
            No visual of the system declares a view.

    Examples:
        >>> from skcomplex.core.viz import primary_view
        >>> from skcomplex.models import create
        >>> view = primary_view(create('gray_scott', shape=(16, 16)))
        >>> type(view).__name__, str(view.origin)
        ('Spatial2DView', 'cells/u')
    """
    spatial, series = _ordered_views(_problem_of(system), parameters=False)
    views = [*spatial, *series]
    if not views:
        error_msg = 'No visual of the system declares a view. Give its element types a visual with `views`.'
        raise ValidationError(error_msg)
    return views[0]


def default_blueprint(
    system: System | Problem,
    *,
    parameters: bool = False,
    primary: str | None = None,
    observables: Sequence[Observable] = (),
) -> rrb.Blueprint:
    """Return a Rerun blueprint built from the views the visuals of a system declare.

    Spatial views come first and side by side, with time series views below them. Scalar elements of one type share a
    time series view with at most six series, and time series views show the whole run. The side panels are collapsed,
    the top panel is hidden, and the time panel follows the latest frame of the simulated time.

    Args:
        system:
            The system, or a problem compiled from one.

        parameters:
            Whether to add a time series view of the parameters.

        primary:
            The name or the origin of the view to put first, with its row, or `None` for `primary_view(system)`.

        observables:
            The observables of the run, which get a time series view, unless `system` is a compiled problem.

    Returns:
        blueprint:
            The blueprint, whose first view is the primary view.

    Raises:
        ValidationError:
            No view has the name or the origin of `primary`.

    Examples:
        >>> from skcomplex.core.viz import default_blueprint
        >>> from skcomplex.models import create
        >>> blueprint = default_blueprint(create('lotka_volterra'))
        >>> [(type(view).__name__, str(view.name)) for view in blueprint.root_container.contents]
        [('TimeSeriesView', 'Population')]
        >>> blueprint = default_blueprint(create('gray_scott', shape=(16, 16)), primary='cells/v')
        >>> [str(view.origin) for view in blueprint.root_container.contents]
        ['cells/v', 'cells/u']
    """
    spatial, series = _ordered_views(_problem_of(system, observables), parameters=parameters)
    ordered = [spatial, series] if primary is None else _with_primary(spatial, series, primary)
    rows = [_row(views) for views in ordered if views]
    if len(rows) > 1:
        shares = [SPATIAL_ROW_SHARE, SERIES_ROW_SHARE] if ordered[0] is spatial else [SERIES_ROW_SHARE] * 2
        root: rrb.View | rrb.Container = rrb.Vertical(*rows, row_shares=shares)
    else:
        root = rows[0] if rows else rrb.Vertical()
    return rrb.Blueprint(
        root,
        rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
        rrb.TopPanel(state='hidden'),
        collapse_panels=True,
    )


def save_blueprint(blueprint: rrb.Blueprint, path: str | Path, *, application_id: str = 'skcomplex') -> Path:
    """Write a blueprint to a `.rbl` file that the Rerun viewer opens with recordings of one application.

    Args:
        blueprint:
            The blueprint.

        path:
            The path of the `.rbl` file.

        application_id:
            The application id of the recordings the blueprint lays out.

    Returns:
        path:
            The path of the written file.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from skcomplex.core.viz import default_blueprint, save_blueprint
        >>> from skcomplex.models import create
        >>> path = save_blueprint(default_blueprint(create('lorenz')), Path(tempfile.mkdtemp()) / 'lorenz.rbl')
        >>> path.suffix, path.stat().st_size > 0
        ('.rbl', True)
    """
    path = Path(path)
    blueprint.save(application_id, path)
    return path
