"""Draw the state of elements as Rerun archetypes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import deque
from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ..base import Domain, Registry, VisualSpec
from ..compile import CompiledAgents, CompiledElement, Problem
from ..runtime import Frame
from ..solvers import AgentSnapshot
from ..space import ContinuousSpace, LatticeSpace, NetworkSpace, Space

MAX_POINT_DIMENSIONS = 3
UPWARD = np.array([1.0, -1.0])
PATH_COLOR = (86, 180, 233)
PATH_WIDTH = 1.0
LAYOUT_SEED = 0
NODE_SPACING = 40.0
POINT_SHARE = 0.02
CROWD_SIZE = 10
CROWD_EXPONENT = 0.25
TRAIL_LENGTH = 256
MAX_TRAIL_POINTS = 10
TRAIL_WIDTH = 0.25
GRID_RADIUS = 0.4
MIN_HISTORY = 64
MAX_HISTORY = 256
SPACE_TIME = 'space_time'
HEXAGON_RADIUS = float(1 / np.sqrt(3.0))
SERIES_COLORS = (
    (230, 159, 0),
    (86, 180, 233),
    (0, 158, 115),
    (240, 228, 66),
    (0, 114, 178),
    (213, 94, 0),
    (204, 121, 167),
    (170, 170, 170),
)
POINT_FLOOR = 0.25
VIEW_MARGIN = 0.05
COLOR_SCALE = np.array(
    [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
    dtype=float,
)


def _colors_of(values: npt.NDArray[np.float64], low: float, high: float) -> npt.NDArray[np.uint8]:
    """Return one color per value on a scale from blue at `low` to yellow at `high`, bright enough to see on black."""
    fractions = np.clip((values - low) / (high - low), 0.0, 1.0) if high > low else np.zeros_like(values)
    fractions = POINT_FLOOR + (1.0 - POINT_FLOOR) * fractions
    anchors = np.linspace(0.0, 1.0, len(COLOR_SCALE))
    channels = [np.interp(fractions, anchors, COLOR_SCALE[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=-1).round().astype(np.uint8)


PALETTE = (
    np.stack(
        [
            np.interp(np.linspace(0.0, 1.0, 256), np.linspace(0.0, 1.0, len(COLOR_SCALE)), COLOR_SCALE[:, channel])
            for channel in range(3)
        ],
        axis=-1,
    )
    .round()
    .astype(np.uint8)
)


def _widened(span: tuple[float, float] | None, values: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Return a color range widened to cover values, starting from their extremes when there is no range yet."""
    low, high = float(np.min(values)), float(np.max(values))
    return (low, high) if span is None else (min(span[0], low), max(span[1], high))


def _palette_colors(values: npt.NDArray[np.float64], low: float, high: float) -> npt.NDArray[np.uint8]:
    """Return the colors of an array of values from a palette of 256 levels, fast enough for large images."""
    if high <= low:
        return np.broadcast_to(PALETTE[0], (*values.shape, 3))
    levels = np.clip((values - low) * (255.0 / (high - low)), 0.0, 255.0).astype(np.uint8)
    return PALETTE[levels]


PHASE_SCALE = np.array(
    [[230, 90, 80], [225, 190, 70], [90, 190, 110], [70, 160, 210], [150, 100, 200], [230, 90, 80]],
    dtype=float,
)
PHASE_PALETTE = (
    np.stack(
        [
            np.interp(np.linspace(0.0, 1.0, 256), np.linspace(0.0, 1.0, len(PHASE_SCALE)), PHASE_SCALE[:, channel])
            for channel in range(3)
        ],
        axis=-1,
    )
    .round()
    .astype(np.uint8)
)
MODULUS = 'modulus'
PHASE = 'phase'
PARTS = ('real', 'imag')


def _phase_colors(phases: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """Return the colors of phases in `(-pi, pi]` from a cyclic palette, so `-pi` and `pi` share one color."""
    levels = np.mod(phases + np.pi, 2 * np.pi) * (255.0 / (2 * np.pi))
    return PHASE_PALETTE[np.clip(levels, 0.0, 255.0).astype(np.uint8)]


def _series_index(problem: Problem, path: str, state: str) -> int:
    """Return the position of a state among the states of every element of a problem, in declaration order."""
    names = [(element.path, name) for element in problem.elements for name in element.states]
    return names.index((path, state)) if (path, state) in names else 0


def _style_series(recording: rr.RecordingStream, path: str, value: object, index: int) -> None:
    """Log static series colors for the components of a value, from the categorical palette starting at an index."""
    values = np.ravel(np.asarray(value))
    parts = [f'{path}/{part}' for part in PARTS] if np.iscomplexobj(values) else [path]
    for offset, entity in enumerate(parts):
        colors = [SERIES_COLORS[(index + offset + column) % len(SERIES_COLORS)] for column in range(values.size)]
        recording.log(entity, rr.SeriesLines(colors=colors), static=True)


def _log_scalars(recording: rr.RecordingStream, path: str, value: object) -> None:
    """Log the components of a value as scalar series, the real and imaginary parts of a complex value apart."""
    values = np.ravel(np.asarray(value))
    if not np.iscomplexobj(values):
        recording.log(path, rr.Scalars(values))
        return
    for part, components in zip(PARTS, (values.real, values.imag), strict=True):
        recording.log(f'{path}/{part}', rr.Scalars(components))


def _radius_of(extent: float, count: int) -> float:
    """Return a point radius that is a share of the extent of a view, smaller for crowds so they do not overlap."""
    return POINT_SHARE * (extent if extent > 0 else 1.0) / max(count / CROWD_SIZE, 1.0) ** CROWD_EXPONENT


def _levels_of(values: npt.NDArray[np.float64], low: float, high: float) -> npt.NDArray[np.uint8]:
    """Return the values as levels from 1 to 255 of a color map, with 0 left for no data."""
    if high <= low:
        return np.ones(values.shape, dtype=np.uint8)
    return (1 + np.clip((values - low) * (254.0 / (high - low)), 0.0, 254.0)).astype(np.uint8)


def _network_layout(graph: nx.Graph) -> dict[str, npt.NDArray[np.float64]]:
    """Return fixed positions of the nodes of a network, with neighbours about a spacing apart.

    The positions are the `pos` attributes of the nodes when every node has one, and a seeded spring layout otherwise.
    """
    scale = NODE_SPACING * float(np.sqrt(max(graph.number_of_nodes(), 1))) / 2
    given = nx.get_node_attributes(graph, 'pos')
    if given and len(given) == graph.number_of_nodes():
        layout = nx.rescale_layout_dict({node: np.asarray(position, dtype=float) for node, position in given.items()})
    else:
        layout = nx.spring_layout(graph, seed=LAYOUT_SEED)
    return {str(node): np.asarray(position, dtype=float) * scale for node, position in layout.items()}


class Visual:
    """Base class of visuals, which log the state of one element to a Rerun recording.

    Subclasses implement `log`, declare the Rerun views that show what they log with `views`, and are registered
    under a name with `register_visual`.
    """

    def views(self, element: CompiledElement | CompiledAgents, problem: Problem) -> list[rrb.View]:
        """Return the Rerun views that show what the visual logs for an element, none by default.

        Args:
            element:
                The compiled element or agents, with their path.

            problem:
                The compiled problem the element belongs to.

        Returns:
            views:
                The views, each with an entity path of the element as origin.
        """
        return []

    def log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log the state of an element at one frame.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path and the dotted names of its variables.

            frame:
                The frame.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError


class ScalarsVisual(Visual):
    """Visual that logs each state component of an element as a time series.

    Every state gets static series colors from a categorical palette, by its position among the states of every element
    of the problem, so the series of one view are told apart. The lag states that keep the past values of a delayed
    state are left out.

    Args:
        components:
            The indices of the components of vector states to log, or `None` for every component.

    Examples:
        >>> from skcomplex.core.viz import ScalarsVisual
        >>> ScalarsVisual()
        ScalarsVisual()
    """

    def __init__(self, components: Sequence[int] | None = None) -> None:
        self.components = components
        self._styled: set[str] = set()

    def __repr__(self) -> str:
        """Return the class name and the chosen components."""
        return 'ScalarsVisual()' if self.components is None else f'ScalarsVisual(components={list(self.components)!r})'

    def _value(self, value: object) -> object:
        """Return the chosen components of a vector value, or the value itself."""
        array = np.ravel(np.asarray(value))
        if self.components is None or array.size == 1:
            return value
        return array[[index for index in self.components if index < array.size]]

    def log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log every state of the element under `<path>/<state>`, a complex one under `real` and `imag` below it.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path and the dotted names of its variables.

            frame:
                The frame.
        """
        for state, name in element.states.items():
            if frame.problem.companions.get(name, ('',))[0] == 'lag':
                continue
            path, value = f'{element.path}/{state}', self._value(frame.value(name))
            if path not in self._styled:
                _style_series(recording, path, value, _series_index(frame.problem, element.path, state))
                self._styled.add(path)
            _log_scalars(recording, path, value)

    def views(self, element: CompiledElement | CompiledAgents, problem: Problem) -> list[rrb.View]:
        """Return one time series view of the states of the element.

        Args:
            element:
                The compiled element, with its path.

            problem:
                The compiled problem the element belongs to.

        Returns:
            views:
                A time series view with the element path as origin.

        Examples:
            >>> from skcomplex.core.compile import compile_system
            >>> from skcomplex.core.viz import ScalarsVisual
            >>> from skcomplex.models import create
            >>> problem = compile_system(create('lotka_volterra'))
            >>> view = ScalarsVisual().views(problem.elements[0], problem)[0]
            >>> type(view).__name__, str(view.origin)
            ('TimeSeriesView', 'prey')
        """
        return [rrb.TimeSeriesView(origin=element.path, name=element.path)]


class PointsVisual(Visual):
    """Visual that logs a vector state as points, one per member of a group, and other states as time series.

    Each state is logged as one batch, so a group of any size takes one log call per state and frame. Positions in
    two dimensions are logged with the second coordinate negated, since Rerun's 2D views point their vertical axis
    down, so the views draw them with the vertical axis up and show the negated value on hover. Without a given
    radius, the points take a share of the extent of the positions logged so far, smaller for larger groups. The
    extent and the count include every element drawn by a points visual of the same position state, so single
    particles that share a space, such as the masses of a chain, take one radius from the whole drawing. A group of
    at most ten points in two or three dimensions also logs the trail of its recent positions as line strips under
    `<path>/trail`, a quarter as wide as that share, whatever the radius.

    Args:
        position:
            The name of the vector state that holds the position.

        radius:
            The radius of the points, or `None` for a share of the extent of the positions.

        color_by:
            The name of a scalar state, or of a vector state whose norm, colors the points, or `None` for one color.

        color_range:
            The values at the two ends of the color scale, or `None` for a range widened to every logged value.

        states:
            The names of the other states to log as time series, or `None` for all of them.

        trail:
            The number of recent positions the trail of each point holds, or 0 for no trail.

    Examples:
        >>> from skcomplex.core.viz import PointsVisual
        >>> PointsVisual(position='x', color_by='v').color_by
        'v'
    """

    def __init__(
        self,
        position: str,
        radius: float | None = None,
        color_by: str | None = None,
        color_range: tuple[float, float] | None = None,
        states: Sequence[str] | None = None,
        *,
        trail: int = TRAIL_LENGTH,
    ) -> None:
        self.position = position
        self.radius = radius
        self.color_by = color_by
        self.color_range = color_range
        self.states = states
        self.trail = trail
        self._trail: deque[npt.NDArray[np.float64]] = deque(maxlen=max(trail, 1))
        self._bounds: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None
        self._span: tuple[float, float] | None = None

    def _colors(self, element: CompiledElement, frame: Frame, count: int) -> npt.NDArray[np.uint8] | None:
        """Return the colors of the points over the given range, or over a range widened to every value so far."""
        if self.color_by is None:
            return None
        values = np.asarray(frame.value(element.states[self.color_by]), dtype=float).reshape(count, -1)
        magnitudes = np.linalg.norm(values, axis=1)
        if self.color_range is not None:
            return _colors_of(magnitudes, *self.color_range)
        self._span = _widened(self._span, magnitudes)
        return _colors_of(magnitudes, *self._span)

    def _extent_radius(self, points: npt.NDArray[np.float64]) -> float:
        """Return a share of the extent of every position logged so far, keeping the bounds of the positions."""
        low, high = points.min(axis=0), points.max(axis=0)
        if self._bounds is not None:
            low, high = np.minimum(low, self._bounds[0]), np.maximum(high, self._bounds[1])
        self._bounds = (low, high)
        extent = float((high - low).max()) or float(np.abs(points).max())
        return _radius_of(extent, len(points))

    def _crowd(self, frame: Frame, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the positions of every element drawn by a points visual of the same position state and dimension."""
        positions = [
            np.atleast_2d(np.asarray(frame.value(other.states[self.position]), dtype=float))
            for other in frame.problem.elements
            if other.visual.kind == 'points'
            and other.visual.options.get('position') == self.position
            and self.position in other.states
        ]
        same = [item for item in positions if item.shape[1] == points.shape[1]]
        return np.vstack(same) if same else points

    def _radius(self, points: npt.NDArray[np.float64]) -> float:
        """Return the given radius, or a share of the extent of every position logged so far."""
        extent_radius = self._extent_radius(points)
        return extent_radius if self.radius is None else float(self.radius)

    def _log_trail(
        self,
        recording: rr.RecordingStream,
        path: str,
        points: npt.NDArray[np.float64],
        colors: npt.NDArray[np.uint8] | None,
    ) -> None:
        """Log the recent positions of each point as one line strip per point, for small groups only."""
        if not self.trail or len(points) > MAX_TRAIL_POINTS:
            return
        if self._trail and self._trail[-1].shape != points.shape:
            self._trail.clear()
        self._trail.append(points)
        strips = list(np.stack(self._trail, axis=1))
        radius = self._extent_radius(points) * TRAIL_WIDTH
        if points.shape[1] == MAX_POINT_DIMENSIONS:
            recording.log(f'{path}/trail', rr.LineStrips3D(strips, radii=radius, colors=colors))
        else:
            recording.log(f'{path}/trail', rr.LineStrips2D(strips, radii=radius, colors=colors))

    def log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log the position under `<path>`, the trail under `<path>/trail`, and the other states under `<path>/<state>`.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path and the dotted names of its variables.

            frame:
                The frame.
        """
        points = np.atleast_2d(np.asarray(frame.value(element.states[self.position]), dtype=float))
        colors = self._colors(element, frame, len(points))
        radius = self._radius(self._crowd(frame, points)) if points.shape[1] > 1 else 0.0
        if points.shape[1] == MAX_POINT_DIMENSIONS:
            recording.log(element.path, rr.Points3D(points, radii=radius, colors=colors))
            self._log_trail(recording, element.path, points, colors)
        elif points.shape[1] == MAX_POINT_DIMENSIONS - 1:
            shown = points * UPWARD
            recording.log(element.path, rr.Points2D(shown, radii=radius, colors=colors))
            self._log_trail(recording, element.path, shown, colors)
        else:
            recording.log(f'{element.path}/{self.position}', rr.Scalars(points.ravel()))
        others = (
            self.states if self.states is not None else [state for state in element.states if state != self.position]
        )
        for state in others:
            _log_scalars(recording, f'{element.path}/{state}', frame.value(element.states[state]))

    def views(self, element: CompiledElement | CompiledAgents, problem: Problem) -> list[rrb.View]:
        """Return a spatial view of the positions by their dimension, and a time series view of the other states.

        Args:
            element:
                The compiled element, with its path and the dotted names of its states.

            problem:
                The compiled problem the element belongs to.

        Returns:
            views:
                A 3D or 2D spatial view with the element path as origin, or a time series view of the position in other
                dimensions, followed by a time series view of the other logged states, if any.

        Examples:
            >>> from skcomplex.core.compile import compile_system
            >>> from skcomplex.core.viz import PointsVisual
            >>> from skcomplex.models import create
            >>> problem = compile_system(create('ideal_gas', n_molecules=10, n_dim=3))
            >>> views = PointsVisual(position='x', states=()).views(problem.elements[0], problem)
            >>> [type(view).__name__ for view in views]
            ['Spatial3DView']
        """
        if not isinstance(element, CompiledElement):
            return []
        dimension = problem.states.locate(element.states[self.position])[0].shape[-1:] or (1,)
        path = element.path
        views: list[rrb.View] = []
        if dimension[0] == MAX_POINT_DIMENSIONS:
            views.append(rrb.Spatial3DView(origin=path, name=path))
        elif dimension[0] == MAX_POINT_DIMENSIONS - 1:
            views.append(rrb.Spatial2DView(origin=path, name=path))
        else:
            views.append(rrb.TimeSeriesView(origin=f'{path}/{self.position}', name=f'{path}/{self.position}'))
        others = (
            self.states if self.states is not None else [state for state in element.states if state != self.position]
        )
        if others:
            contents = [f'+ {path}/{state}/**' for state in others]
            views.append(rrb.TimeSeriesView(origin=path, contents=contents, name=path))
        return views


class AgentsVisual(Visual):
    """Visual that logs the agents of a group as one batch per frame, colored by a state.

    Agents on a grid or in a continuous space are logged as `Points2D` at their cell centres or positions, and agents
    on a network as `GraphNodes` at their nodes, which keep the `pos` attributes of the nodes, or the positions of a
    seeded spring layout of the network. On a grid the radius is in cells, and on a network a fraction of the spacing
    of neighbouring nodes. Without a given radius, agents in a continuous space take a share of the side of the space,
    smaller for larger groups. Agents in a continuous space are logged with their second coordinate negated, so the
    view, bounded by the space, draws them with the vertical axis up. Every point carries the agent id as a hidden
    label, so agents keep their identity across
    frames and removed agents disappear. Agents whose position is a state are logged at that state: positions of one
    coordinate, such as cars on a road, as one thin series of one color per agent under `<path>/<position>`, so their
    time series view, without a legend, is a space-time diagram, and positions of two coordinates as `Points2D` with
    the vertical axis up.

    Args:
        color_by:
            The name of a numeric state that colors the agents, or `None` for one color.

        color_range:
            The values at the two ends of the color scale, or `None` for a range widened to every logged value.

        radius:
            The radius of the points, or `None` for 0.4 on grids and networks and a share of the side of other spaces.

        position:
            The name of a state that holds the position of every agent, or `None` for the location in the space.

    Examples:
        >>> from skcomplex.core.viz import AgentsVisual
        >>> AgentsVisual(color_by='opinion').color_by
        'opinion'
    """

    def __init__(
        self,
        color_by: str | None = None,
        color_range: tuple[float, float] | None = None,
        radius: float | None = None,
        *,
        position: str | None = None,
    ) -> None:
        self.color_by = color_by
        self.color_range = color_range
        self.radius = radius
        self.position = position
        self._styled: set[str] = set()
        self._layouts: dict[int, dict[str, npt.NDArray[np.float64]]] = {}
        self._span: tuple[float, float] | None = None

    def _layout(self, space: NetworkSpace) -> dict[str, npt.NDArray[np.float64]]:
        """Return the positions of the nodes of a network, computed once per network."""
        if id(space.graph) not in self._layouts:
            self._layouts[id(space.graph)] = _network_layout(space.graph)
        return self._layouts[id(space.graph)]

    def _colors(self, snapshot: AgentSnapshot) -> npt.NDArray[np.uint8] | None:
        """Return one color per agent over the given range, or over a range widened to every value so far."""
        if self.color_by is None or not len(snapshot):
            return None
        values = np.asarray(snapshot.values[self.color_by], dtype=float)
        if self.color_range is not None:
            return _colors_of(values, *self.color_range)
        self._span = _widened(self._span, values)
        return _colors_of(values, *self._span)

    def _radius(self, space: Space | None, positions: npt.NDArray[np.float64]) -> float:
        """Return the given radius, a radius in cells on grids and networks, or a share of the side of other spaces."""
        if self.radius is not None:
            return float(self.radius)
        if space is None or isinstance(space, LatticeSpace | NetworkSpace):
            return GRID_RADIUS
        if isinstance(space, ContinuousSpace):
            return _radius_of(float(max(space.width, space.height)), len(positions))
        extent = float(np.ptp(positions, axis=0).max()) if len(positions) else 0.0
        return _radius_of(extent, len(positions))

    def log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Raise an error, because agents are logged with `log_agents`.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element.

            frame:
                The frame.

        Raises:
            NotImplementedError:
                Agents are not compiled elements.
        """
        raise NotImplementedError

    def log_agents(
        self,
        recording: rr.RecordingStream,
        path: str,
        snapshot: AgentSnapshot,
        space: Space | None,
    ) -> None:
        """Log the agents of a group at one frame.

        Args:
            recording:
                The Rerun recording.

            path:
                The path of the agent group.

            snapshot:
                The agents alive at the frame.

            space:
                The space the agents live in, or `None`.
        """
        colors, labels = self._colors(snapshot), [str(agent) for agent in snapshot.ids]
        if self.position is not None:
            positions = np.asarray(snapshot.values[self.position], dtype=float).reshape(len(snapshot), -1)
            if positions.shape[1] == 1:
                entity = f'{path}/{self.position}'
                if entity not in self._styled:
                    style = rr.SeriesLines(colors=[PATH_COLOR] * len(positions), widths=PATH_WIDTH)
                    recording.log(entity, style, static=True)
                    self._styled.add(entity)
                recording.log(entity, rr.Scalars(positions[:, 0]))
                return
            positions = positions[:, :2] * UPWARD
            extent = float(np.ptp(positions, axis=0).max()) if len(positions) else 0.0
            radius = _radius_of(extent, len(positions)) if self.radius is None else float(self.radius)
            recording.log(path, rr.Points2D(positions, colors=colors, radii=radius, labels=labels, show_labels=False))
            return
        if snapshot.locations is None:
            return
        if isinstance(space, NetworkSpace):
            layout = self._layout(space)
            nodes = [str(node) for node in np.ravel(snapshot.locations).tolist()]
            recording.log(
                path,
                rr.GraphNodes(
                    nodes,
                    positions=[layout[node] for node in nodes],
                    colors=colors,
                    labels=labels,
                    show_labels=False,
                    radii=self._radius(space, np.zeros((0, 2))) * NODE_SPACING / 2,
                ),
            )
            return
        positions = np.asarray(snapshot.locations, dtype=float).reshape(len(snapshot), -1)
        if isinstance(space, LatticeSpace):
            positions = positions + 0.5
        if isinstance(space, ContinuousSpace):
            positions = positions * UPWARD
        radius = self._radius(space, positions)
        recording.log(path, rr.Points2D(positions, colors=colors, radii=radius, labels=labels, show_labels=False))

    def views(self, element: CompiledElement | CompiledAgents, problem: Problem) -> list[rrb.View]:
        """Return a graph view of the whole network, a 2D spatial view for agents in a space, and none without one.

        Args:
            element:
                The compiled agents, with their path.

            problem:
                The compiled problem, with the space the agents live in.

        Returns:
            views:
                One view with the path of the agents as origin, or none for agents without a location or a
                position state.

        Examples:
            >>> from skcomplex.core.compile import compile_system
            >>> from skcomplex.core.viz import AgentsVisual
            >>> from skcomplex.models import create
            >>> problem = compile_system(create('voter_model'))
            >>> type(AgentsVisual().views(problem.agents[0], problem)[0]).__name__
            'GraphView'
        """
        if self.position is not None and isinstance(element, CompiledAgents):
            initial: Any = element.values[self.position]
            if np.ndim(initial) < 2:  # noqa: PLR2004
                origin = f'{element.path}/{self.position}'
                return [rrb.TimeSeriesView(origin=origin, name=origin, plot_legend=rrb.PlotLegend(visible=False))]
            return [rrb.Spatial2DView(origin=element.path, name=element.path)]
        if problem.agent_space is None:
            return []
        if isinstance(problem.agent_space, NetworkSpace):
            positions = np.array([[0.0, 0.0], *self._layout(problem.agent_space).values()])
            low, high = positions.min(axis=0) - NODE_SPACING, positions.max(axis=0) + NODE_SPACING
            bounds = rrb.VisualBounds2D(x_range=(low[0], high[0]), y_range=(low[1], high[1]))
            return [rrb.GraphView(origin=element.path, name=element.path, visual_bounds=bounds)]
        space = problem.agent_space
        if isinstance(space, ContinuousSpace):
            margin = VIEW_MARGIN * max(space.width, space.height)
            bounds = rrb.VisualBounds2D(
                x_range=(-margin, space.width + margin),
                y_range=(-space.height - margin, margin),
            )
            return [rrb.Spatial2DView(origin=element.path, name=element.path, visual_bounds=bounds)]
        return [rrb.Spatial2DView(origin=element.path, name=element.path)]


class ImageVisual(Visual):
    """Visual that logs every state of a lattice group as an image colored by value, or a tensor in 3D and more.

    A one-dimensional lattice also logs a space-time image under `<path>/space_time/<state>`, with one row per logged
    frame, the oldest at the top, and at most `history` rows. Its colors span the values of the rows it shows.

    Args:
        states:
            The states to log, or `None` for every state.

        low:
            The value of the darkest color, or `None` for the smallest value of every frame.

        high:
            The value of the brightest color, or `None` for the largest value of every frame.

        history:
            The largest number of rows of the space-time image of a one-dimensional lattice, 0 for no space-time
            image, or `None` for as many rows as the lattice has sites, between 64 and 256.

    Examples:
        >>> from skcomplex.core.viz import ImageVisual
        >>> ImageVisual(low=0, high=1)
        ImageVisual(states=None, low=0, high=1, history=None)
    """

    def __init__(
        self,
        states: Sequence[str] | None = None,
        low: float | None = None,
        high: float | None = None,
        history: int | None = None,
    ) -> None:
        self.states = states
        self.low = low
        self.high = high
        self.history = history
        self._rows: dict[str, deque[npt.NDArray[np.float64]]] = {}

    def __repr__(self) -> str:
        """Return the options."""
        return f'ImageVisual(states={self.states!r}, low={self.low!r}, high={self.high!r}, history={self.history!r})'

    def _history_of(self, sites: int) -> int:
        """Return the number of rows of the space-time image of a one-dimensional lattice of a number of sites."""
        if self.history is not None:
            return self.history
        return int(np.clip(sites, MIN_HISTORY, MAX_HISTORY))

    def _range(self, values: npt.NDArray[np.float64]) -> tuple[float, float]:
        """Return the values of the two ends of the color scale, the given ones or the extremes of the values."""
        low = float(values.min()) if self.low is None else float(self.low)
        high = float(values.max()) if self.high is None else float(self.high)
        return low, high

    def _log_space_time(self, recording: rr.RecordingStream, path: str, values: npt.NDArray[np.float64]) -> None:
        """Log the recent rows of a one-dimensional lattice as a space-time depth image with a color map."""
        rows = self._history_of(len(values))
        if not rows:
            return
        history = self._rows.setdefault(path, deque(maxlen=rows))
        history.append(values)
        image = np.stack(history)
        depth = rr.DepthImage(_levels_of(image, *self._range(image)), colormap='viridis', depth_range=(0.0, 255.0))
        recording.log(path, depth)

    def _log_image(
        self,
        recording: rr.RecordingStream,
        path: str,
        values: npt.NDArray[np.float64],
        *,
        cyclic: bool,
    ) -> None:
        """Log lattice values as an image colored by value or by a cyclic palette, or as a tensor in 3D and more."""
        if values.ndim > 2:  # noqa: PLR2004
            recording.log(path, rr.Tensor(values))
            return
        values = np.atleast_2d(values)
        colors = _phase_colors(values) if cyclic else _palette_colors(values, *self._range(values))
        height, width = values.shape
        recording.log(
            path,
            rr.Image(bytes=colors.tobytes(), width=width, height=height, color_model='RGB', datatype='U8'),
        )

    def log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log every chosen state of a lattice group under `<path>/<state>`, and a space-time image in one dimension.

        A complex state logs its modulus under `<path>/<state>/modulus`, colored over the range, and its phase under
        `<path>/<state>/phase`, colored by a cyclic palette over `(-pi, pi]`. An angle state is colored by the same
        cyclic palette. Its space-time image shows the modulus.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path and the dotted names of its variables.

            frame:
                The frame.
        """
        for state, name in element.states.items():
            if self.states is not None and state not in self.states:
                continue
            values = np.asarray(frame.lattice(name))
            path = f'{element.path}/{state}'
            magnitudes = np.abs(values) if np.iscomplexobj(values) else np.asarray(values, dtype=float)
            if magnitudes.ndim == 1:
                self._log_space_time(recording, f'{element.path}/{SPACE_TIME}/{state}', magnitudes)
            if not np.iscomplexobj(values):
                cyclic = frame.problem.states.entry(name).domain is Domain.ANGLE
                self._log_image(recording, path, magnitudes, cyclic=cyclic)
                continue
            self._log_image(recording, f'{path}/{MODULUS}', magnitudes, cyclic=False)
            self._log_image(recording, f'{path}/{PHASE}', np.angle(values), cyclic=True)

    def views(self, element: CompiledElement | CompiledAgents, problem: Problem) -> list[rrb.View]:
        """Return one 2D spatial view per chosen state, or a tensor view for a lattice of three or more dimensions.

        A one-dimensional lattice gets the 2D spatial views of its space-time images first, named `<state> history`.

        Args:
            element:
                The compiled lattice group, with its path and the dotted names of its states.

            problem:
                The compiled problem, with the lattice shape of every state.

        Returns:
            views:
                One view per chosen state with `<path>/<state>` as origin, two for a complex state with its modulus
                and phase as origins, after one view per state with `<path>/space_time/<state>` as origin in one
                dimension.

        Examples:
            >>> from skcomplex.core.compile import compile_system
            >>> from skcomplex.core.viz import ImageVisual
            >>> from skcomplex.models import create
            >>> problem = compile_system(create('gray_scott', shape=(16, 16)))
            >>> [str(view.origin) for view in ImageVisual().views(problem.elements[0], problem)]
            ['cells/u', 'cells/v']
            >>> problem = compile_system(create('elementary_automaton'))
            >>> [str(view.origin) for view in ImageVisual().views(problem.elements[0], problem)]
            ['cells/space_time/alive', 'cells/alive']
        """
        if not isinstance(element, CompiledElement):
            return []
        histories: list[rrb.View] = []
        views: list[rrb.View] = []
        for state, name in element.states.items():
            if self.states is not None and state not in self.states:
                continue
            path, shape = f'{element.path}/{state}', problem.lattice_shape(name) or ()
            if len(shape) == 1 and self._history_of(shape[0]):
                histories.append(
                    rrb.Spatial2DView(origin=f'{element.path}/{SPACE_TIME}/{state}', name=f'{state} history'),
                )
            kind = rrb.TensorView if len(shape) > 2 else rrb.Spatial2DView  # noqa: PLR2004
            if problem.states.entry(name).is_complex:
                views += [kind(origin=f'{path}/{part}', name=f'{state} {part}') for part in (MODULUS, PHASE)]
            else:
                views.append(kind(origin=path, name=state))
        return [*histories, *views]


class SiteVisual(Visual):
    """Visual that logs every state of a lattice group that is not square as colored sites.

    A triangular lattice is logged as `Points2D` at the site positions, with radii that cover the hexagonal cells of
    the sites. A lattice group on a network is logged as `GraphNodes` colored by state, at the `pos` attributes of the
    nodes or a seeded spring layout, with its `GraphEdges` logged once as static data.

    Args:
        states:
            The states to log, or `None` for every state.

        low:
            The value of the darkest color, or `None` for the smallest value of every frame.

        high:
            The value of the brightest color, or `None` for the largest value of every frame.

    Examples:
        >>> from skcomplex.core.viz import SiteVisual
        >>> SiteVisual(low=0, high=2)
        SiteVisual(states=None, low=0, high=2)
    """

    def __init__(
        self,
        states: Sequence[str] | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        self.states = states
        self.low = low
        self.high = high
        self._layouts: dict[int, dict[str, npt.NDArray[np.float64]]] = {}
        self._edges: set[str] = set()

    def __repr__(self) -> str:
        """Return the options."""
        return f'SiteVisual(states={self.states!r}, low={self.low!r}, high={self.high!r})'

    def _colors(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
        """Return the colors of site values over the given range or the extremes of the values."""
        low = float(values.min()) if self.low is None else float(self.low)
        high = float(values.max()) if self.high is None else float(self.high)
        return _palette_colors(values, low, high)

    def _layout(self, space: NetworkSpace) -> dict[str, npt.NDArray[np.float64]]:
        """Return the positions of the nodes of a network, computed once per network."""
        if id(space.graph) not in self._layouts:
            self._layouts[id(space.graph)] = _network_layout(space.graph)
        return self._layouts[id(space.graph)]

    def _log_graph(
        self,
        recording: rr.RecordingStream,
        path: str,
        space: NetworkSpace,
        values: npt.NDArray[np.float64],
    ) -> None:
        """Log the nodes of a network colored by value, and its edges once."""
        layout = self._layout(space)
        nodes = [str(node) for node in space.graph.nodes]
        if path not in self._edges:
            edges = [(str(source), str(target)) for source, target in space.graph.edges]
            recording.log(path, rr.GraphEdges(edges), static=True)
            self._edges.add(path)
        positions = [layout[node] for node in nodes]
        radius = GRID_RADIUS * NODE_SPACING / 2
        drawn = rr.GraphNodes(nodes, positions=positions, colors=self._colors(values), radii=radius, show_labels=False)
        recording.log(path, drawn)

    def log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log every chosen state of a triangular lattice or a network under `<path>/<state>`.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled lattice group, with its path and the dotted names of its states.

            frame:
                The frame.
        """
        space = getattr(element.element, 'space', None)
        for state, name in element.states.items():
            if self.states is not None and state not in self.states:
                continue
            values = np.ravel(np.asarray(frame.value(name), dtype=float))
            path = f'{element.path}/{state}'
            if isinstance(space, NetworkSpace):
                self._log_graph(recording, path, space, values)
                continue
            lattice = space if isinstance(space, LatticeSpace) else LatticeSpace(values.size)
            positions = lattice.positions
            if positions.shape[1] == 1:
                positions = np.column_stack([positions[:, 0], np.zeros(len(positions))])
            radius = HEXAGON_RADIUS * min(lattice.spacings)
            recording.log(path, rr.Points2D(positions, colors=self._colors(values), radii=radius))

    def views(self, element: CompiledElement | CompiledAgents, problem: Problem) -> list[rrb.View]:
        """Return one graph view per chosen state of a lattice group on a network, and a 2D spatial view otherwise.

        Args:
            element:
                The compiled lattice group, with its path and the dotted names of its states.

            problem:
                The compiled problem the group belongs to.

        Returns:
            views:
                One view per chosen state, with `<path>/<state>` as origin.

        Examples:
            >>> import networkx as nx
            >>> from skcomplex.core.base import state
            >>> from skcomplex.core.compile import compile_system
            >>> from skcomplex.core.components import Element
            >>> from skcomplex.core.space import NetworkSpace
            >>> from skcomplex.core.system import System
            >>> from skcomplex.core.viz import SiteVisual
            >>> Node = Element.define('Node', states={'value': state(0.0)})
            >>> system = System(components=[Node.lattice('nodes', NetworkSpace(nx.cycle_graph(4)))])
            >>> problem = compile_system(system)
            >>> [type(view).__name__ for view in SiteVisual().views(problem.elements[0], problem)]
            ['GraphView']
        """
        if not isinstance(element, CompiledElement):
            return []
        space = getattr(element.element, 'space', None)
        states = [state for state in element.states if self.states is None or state in self.states]
        if not isinstance(space, NetworkSpace):
            return [rrb.Spatial2DView(origin=f'{element.path}/{state}', name=state) for state in states]
        positions = np.array(list(self._layout(space).values()))
        low, high = positions.min(axis=0) - NODE_SPACING, positions.max(axis=0) + NODE_SPACING
        bounds = rrb.VisualBounds2D(x_range=(low[0], high[0]), y_range=(low[1], high[1]))
        return [rrb.GraphView(origin=f'{element.path}/{state}', name=state, visual_bounds=bounds) for state in states]


class MeshVisual(Visual):
    """Visual that logs every state of a finite element field as a triangle mesh colored by value.

    Triangle meshes are drawn flat with the value as their height, and tetrahedral meshes as points.

    Args:
        states:
            The states to log, or `None` for every state.

        low:
            The value of the darkest color, or `None` for the smallest value of every frame.

        high:
            The value of the brightest color, or `None` for the largest value of every frame.

    Examples:
        >>> from skcomplex.core.viz import MeshVisual
        >>> MeshVisual(low=0.0)
        MeshVisual(states=None, low=0.0, high=None)
    """

    def __init__(
        self,
        states: Sequence[str] | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        self.states = states
        self.low = low
        self.high = high

    def __repr__(self) -> str:
        """Return the options."""
        return f'MeshVisual(states={self.states!r}, low={self.low!r}, high={self.high!r})'

    def _colors(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
        """Return the colors of node values over the given range or the extremes of the values."""
        low = float(values.min()) if self.low is None else float(self.low)
        high = float(values.max()) if self.high is None else float(self.high)
        return _palette_colors(values, low, high)

    @staticmethod
    def _log_mesh(
        recording: rr.RecordingStream,
        path: str,
        mesh: Any,  # noqa: ANN401
        values: npt.NDArray[np.float64],
        colors: npt.NDArray[np.uint8],
    ) -> None:
        """Log node values as a flat triangle mesh with the values as height, or as points of a tetrahedral mesh."""
        nodes, cells = mesh.nodes, np.asarray(mesh.cells)
        if mesh.n_dim != 2:  # noqa: PLR2004
            recording.log(path, rr.Points3D(nodes, colors=colors))
            return
        positions = np.column_stack([nodes, values])
        recording.log(path, rr.Mesh3D(vertex_positions=positions, triangle_indices=cells.T, vertex_colors=colors))

    def log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log every chosen state of a finite element field under `<path>/<state>`, a complex one as modulus and phase.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path and the dotted names of its variables.

            frame:
                The frame.
        """
        mesh: Any = getattr(element.element, 'space', None)
        for state, name in element.states.items():
            if self.states is not None and state not in self.states:
                continue
            values = np.ravel(frame.value(name))
            path = f'{element.path}/{state}'
            if not np.iscomplexobj(values):
                self._log_mesh(recording, path, mesh, values, self._colors(values))
                continue
            modulus, phase = np.abs(values), np.angle(values)
            self._log_mesh(recording, f'{path}/{MODULUS}', mesh, modulus, self._colors(modulus))
            self._log_mesh(recording, f'{path}/{PHASE}', mesh, phase, _phase_colors(phase))

    def views(self, element: CompiledElement | CompiledAgents, problem: Problem) -> list[rrb.View]:
        """Return one 3D spatial view per chosen state, since meshes are drawn with the value as their height.

        Args:
            element:
                The compiled finite element field, with its path and the dotted names of its states.

            problem:
                The compiled problem the field belongs to.

        Returns:
            views:
                One 3D spatial view per chosen state, with `<path>/<state>` as origin.

        Examples:
            >>> from skcomplex.core.compile import compile_system
            >>> from skcomplex.core.viz import MeshVisual
            >>> from skcomplex.models import create
            >>> problem = compile_system(create('heat_l_shape', refine=1))
            >>> [type(view).__name__ for view in MeshVisual().views(problem.elements[0], problem)]
            ['Spatial3DView']
        """
        if not isinstance(element, CompiledElement):
            return []
        views: list[rrb.View] = []
        for state, name in element.states.items():
            if self.states is not None and state not in self.states:
                continue
            path = f'{element.path}/{state}'
            if problem.states.entry(name).is_complex:
                views += [
                    rrb.Spatial3DView(origin=f'{path}/{part}', name=f'{state} {part}') for part in (MODULUS, PHASE)
                ]
            else:
                views.append(rrb.Spatial3DView(origin=path, name=state))
        return views


VISUALS = Registry[type[Visual]]('visual')


def register_visual(kind: str, visual: type[Visual]) -> None:
    """Register a visual class under the name element types refer to.

    Args:
        kind:
            The name used in `VisualSpec.kind`.

        visual:
            The visual class, built with the options of the `VisualSpec`.
    """
    VISUALS.register(kind, visual)


def resolve_visual(spec: VisualSpec) -> Visual:
    """Build the visual a spec names.

    Args:
        spec:
            The visual spec of an element type.

    Returns:
        visual:
            The visual built with the spec's options.
    """
    options: dict[str, Any] = dict(spec.options)
    return VISUALS.resolve(spec.kind)(**options)


register_visual('scalars', ScalarsVisual)
register_visual('points', PointsVisual)
register_visual('agents', AgentsVisual)
register_visual('image', ImageVisual)
register_visual('sites', SiteVisual)
register_visual('mesh', MeshVisual)
