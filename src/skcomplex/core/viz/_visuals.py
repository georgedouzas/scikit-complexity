"""Draw the state of elements as Rerun archetypes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ..base import Registry, VisualSpec
from ..compile import CompiledAgents, CompiledElement, Problem
from ..model import GridSpace, NetworkSpace, Space
from ..runtime import Frame
from ..solvers import AgentSnapshot

MAX_POINT_DIMENSIONS = 3
LAYOUT_SEED = 0
NODE_SPACING = 40.0
COLOR_SCALE = np.array(
    [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
    dtype=float,
)


def _colors_of(values: npt.NDArray[np.float64], low: float, high: float) -> npt.NDArray[np.uint8]:
    """Return one color per value on a fixed scale from dark purple at `low` to yellow at `high`."""
    fractions = np.clip((values - low) / (high - low), 0.0, 1.0) if high > low else np.zeros_like(values)
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


def _palette_colors(values: npt.NDArray[np.float64], low: float, high: float) -> npt.NDArray[np.uint8]:
    """Return the colors of an array of values from a palette of 256 levels, fast enough for large images."""
    if high <= low:
        return np.broadcast_to(PALETTE[0], (*values.shape, 3))
    levels = np.clip((values - low) * (255.0 / (high - low)), 0.0, 255.0).astype(np.uint8)
    return PALETTE[levels]


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

    Examples:
        >>> from skcomplex.core.viz import ScalarsVisual
        >>> ScalarsVisual()
        ScalarsVisual()
    """

    def __repr__(self) -> str:
        """Return the class name."""
        return 'ScalarsVisual()'

    def log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log every state of the element under `<path>/<state>`.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path and the dotted names of its variables.

            frame:
                The frame.
        """
        for state, name in element.states.items():
            recording.log(f'{element.path}/{state}', rr.Scalars(np.ravel(frame.value(name))))

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

    Each state is logged as one batch, so a group of any size takes one log call per state and frame.

    Args:
        position:
            The name of the vector state that holds the position.

        radius:
            The radius of the points.

        color_by:
            The name of a scalar state, or of a vector state whose norm, colors the points, or `None` for one color.

        color_range:
            The values at the two ends of the color scale, or `None` to fix them from the first logged frame.

        states:
            The names of the other states to log as time series, or `None` for all of them.

    Examples:
        >>> from skcomplex.core.viz import PointsVisual
        >>> PointsVisual(position='x', color_by='v').color_by
        'v'
    """

    def __init__(
        self,
        position: str,
        radius: float = 0.05,
        color_by: str | None = None,
        color_range: tuple[float, float] | None = None,
        states: Sequence[str] | None = None,
    ) -> None:
        self.position = position
        self.radius = radius
        self.color_by = color_by
        self.color_range = color_range
        self.states = states

    def _colors(self, element: CompiledElement, frame: Frame, count: int) -> npt.NDArray[np.uint8] | None:
        """Return the colors of the points, fixing the color range on first use when it is not given."""
        if self.color_by is None:
            return None
        values = np.asarray(frame.value(element.states[self.color_by]), dtype=float).reshape(count, -1)
        magnitudes = np.linalg.norm(values, axis=1)
        if self.color_range is None:
            self.color_range = (float(magnitudes.min()), float(magnitudes.max()))
        return _colors_of(magnitudes, *self.color_range)

    def log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log the position under `<path>` and the other states under `<path>/<state>`.

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
        if points.shape[1] == MAX_POINT_DIMENSIONS:
            recording.log(element.path, rr.Points3D(points, radii=self.radius, colors=colors))
        elif points.shape[1] == MAX_POINT_DIMENSIONS - 1:
            recording.log(element.path, rr.Points2D(points, radii=self.radius, colors=colors))
        else:
            recording.log(f'{element.path}/{self.position}', rr.Scalars(points.ravel()))
        others = (
            self.states if self.states is not None else [state for state in element.states if state != self.position]
        )
        for state in others:
            recording.log(f'{element.path}/{state}', rr.Scalars(np.ravel(frame.value(element.states[state]))))

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
    seeded spring layout of the network. On a network the radius is a fraction of the spacing of neighbouring nodes.
    Every point carries the agent id as a hidden label, so agents keep their identity across frames and removed agents
    disappear.

    Args:
        color_by:
            The name of a numeric state that colors the agents, or `None` for one color.

        color_range:
            The values at the two ends of the color scale, or `None` to fix them from the first logged frame.

        radius:
            The radius of the points.

    Examples:
        >>> from skcomplex.core.viz import AgentsVisual
        >>> AgentsVisual(color_by='opinion').color_by
        'opinion'
    """

    def __init__(
        self,
        color_by: str | None = None,
        color_range: tuple[float, float] | None = None,
        radius: float = 0.4,
    ) -> None:
        self.color_by = color_by
        self.color_range = color_range
        self.radius = radius
        self._layouts: dict[int, dict[str, npt.NDArray[np.float64]]] = {}

    def _layout(self, space: NetworkSpace) -> dict[str, npt.NDArray[np.float64]]:
        """Return the positions of the nodes of a network, computed once per network."""
        if id(space.graph) not in self._layouts:
            self._layouts[id(space.graph)] = _network_layout(space.graph)
        return self._layouts[id(space.graph)]

    def _colors(self, snapshot: AgentSnapshot) -> npt.NDArray[np.uint8] | None:
        """Return one color per agent from the colored state, fixing the range on first use."""
        if self.color_by is None or not len(snapshot):
            return None
        values = np.asarray(snapshot.values[self.color_by], dtype=float)
        if self.color_range is None:
            self.color_range = (float(values.min()), float(values.max()))
        return _colors_of(values, *self.color_range)

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
        if snapshot.locations is None:
            return
        colors, labels = self._colors(snapshot), [str(agent) for agent in snapshot.ids]
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
                    radii=self.radius * NODE_SPACING / 2,
                ),
            )
            return
        positions = np.asarray(snapshot.locations, dtype=float).reshape(len(snapshot), -1)
        if isinstance(space, GridSpace):
            positions = positions + 0.5
        recording.log(path, rr.Points2D(positions, colors=colors, radii=self.radius, labels=labels, show_labels=False))

    def views(self, element: CompiledElement | CompiledAgents, problem: Problem) -> list[rrb.View]:
        """Return a graph view of the whole network, a 2D spatial view for agents in a space, and none without one.

        Args:
            element:
                The compiled agents, with their path.

            problem:
                The compiled problem, with the space the agents live in.

        Returns:
            views:
                One view with the path of the agents as origin, or none for agents without a location.

        Examples:
            >>> from skcomplex.core.compile import compile_system
            >>> from skcomplex.core.viz import AgentsVisual
            >>> from skcomplex.models import create
            >>> problem = compile_system(create('voter_model'))
            >>> type(AgentsVisual().views(problem.agents[0], problem)[0]).__name__
            'GraphView'
        """
        if problem.agent_space is None:
            return []
        if isinstance(problem.agent_space, NetworkSpace):
            positions = np.array([[0.0, 0.0], *self._layout(problem.agent_space).values()])
            low, high = positions.min(axis=0) - NODE_SPACING, positions.max(axis=0) + NODE_SPACING
            bounds = rrb.VisualBounds2D(x_range=(low[0], high[0]), y_range=(low[1], high[1]))
            return [rrb.GraphView(origin=element.path, name=element.path, visual_bounds=bounds)]
        return [rrb.Spatial2DView(origin=element.path, name=element.path)]


class ImageVisual(Visual):
    """Visual that logs every state of a lattice group as an image colored by value, or a tensor in 3D and more.

    Args:
        states:
            The states to log, or `None` for every state.

        low:
            The value of the darkest color, or `None` for the smallest value of every frame.

        high:
            The value of the brightest color, or `None` for the largest value of every frame.

    Examples:
        >>> from skcomplex.core.viz import ImageVisual
        >>> ImageVisual(low=0, high=1)
        ImageVisual(states=None, low=0, high=1)
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
        return f'ImageVisual(states={self.states!r}, low={self.low!r}, high={self.high!r})'

    def log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log every chosen state of a lattice group under `<path>/<state>`.

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
            values = np.asarray(frame.lattice(name), dtype=float)
            path = f'{element.path}/{state}'
            if values.ndim > 2:  # noqa: PLR2004
                recording.log(path, rr.Tensor(values))
                continue
            values = np.atleast_2d(values)
            low = float(values.min()) if self.low is None else float(self.low)
            high = float(values.max()) if self.high is None else float(self.high)
            recording.log(path, rr.Image(_palette_colors(values, low, high)))

    def views(self, element: CompiledElement | CompiledAgents, problem: Problem) -> list[rrb.View]:
        """Return one 2D spatial view per chosen state, or a tensor view for a lattice of three or more dimensions.

        Args:
            element:
                The compiled lattice group, with its path and the dotted names of its states.

            problem:
                The compiled problem, with the lattice shape of every state.

        Returns:
            views:
                One view per chosen state, with `<path>/<state>` as origin.

        Examples:
            >>> from skcomplex.core.compile import compile_system
            >>> from skcomplex.core.viz import ImageVisual
            >>> from skcomplex.models import create
            >>> problem = compile_system(create('gray_scott', shape=(16, 16)))
            >>> [str(view.origin) for view in ImageVisual().views(problem.elements[0], problem)]
            ['cells/u', 'cells/v']
        """
        if not isinstance(element, CompiledElement):
            return []
        views: list[rrb.View] = []
        for state, name in element.states.items():
            if self.states is not None and state not in self.states:
                continue
            path, shape = f'{element.path}/{state}', problem.lattice_shape(name) or ()
            kind = rrb.TensorView if len(shape) > 2 else rrb.Spatial2DView  # noqa: PLR2004
            views.append(kind(origin=path, name=state))
        return views


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

    def log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log every chosen state of a finite element field under `<path>/<state>`.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path and the dotted names of its variables.

            frame:
                The frame.
        """
        mesh: Any = getattr(element.element, 'space', None)
        nodes, cells = mesh.nodes, np.asarray(mesh.cells)
        for state, name in element.states.items():
            if self.states is not None and state not in self.states:
                continue
            values = np.ravel(frame.value(name))
            low = float(values.min()) if self.low is None else float(self.low)
            high = float(values.max()) if self.high is None else float(self.high)
            colors = _palette_colors(values, low, high)
            path = f'{element.path}/{state}'
            if mesh.n_dim == 2:  # noqa: PLR2004
                positions = np.column_stack([nodes, values])
                recording.log(
                    path,
                    rr.Mesh3D(vertex_positions=positions, triangle_indices=cells.T, vertex_colors=colors),
                )
            else:
                recording.log(path, rr.Points3D(nodes, colors=colors))

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
        states = [state for state in element.states if self.states is None or state in self.states]
        return [rrb.Spatial3DView(origin=f'{element.path}/{state}', name=state) for state in states]


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
register_visual('mesh', MeshVisual)
