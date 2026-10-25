"""Draw the agents of a group as points or network nodes colored by a state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ...core.space import BaseSpace, ContinuousSpace, LatticeSpace, NetworkSpace
from ..compile import CompiledAgents, CompiledElement, Problem
from ..runtime import Frame
from ..solvers import AgentSnapshot
from ._colors import _widened, colors_of
from ._utils import GRID_RADIUS, NODE_SPACING, UPWARD, _network_layout, _radius_of
from ._visuals import BaseVisual, register_visual

PATH_COLOR = (86, 180, 233)
PATH_WIDTH = 1.0
VIEW_MARGIN = 0.05


class AgentsVisual(BaseVisual):
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
        >>> from skcomplex.engine.viz import AgentsVisual
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
            return colors_of(values, *self.color_range)
        self._span = _widened(self._span, values)
        return colors_of(values, *self._span)

    def _radius(self, space: BaseSpace | None, positions: npt.NDArray[np.float64]) -> float:
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
        space: BaseSpace | None,
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
            >>> from skcomplex.engine.compile import compile_system
            >>> from skcomplex.engine.viz import AgentsVisual
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


register_visual('agents', AgentsVisual)
