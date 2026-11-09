"""Draw the agents of a group as points or network nodes colored by a state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ...components.spaces import BaseSpace, ContinuousSpace, LatticeSpace, NetworkSpace
from ...components.specs import CompiledAgentsSpec, CompiledElementSpec
from ...specs import AgentSnapshotSpec
from ..problems import BaseProblemSpec
from ._base import GRID_RADIUS, NODE_SPACING, UPWARD, BaseVisual

PATH_COLOR = (86, 180, 233)
PATH_WIDTH = 1.0
VIEW_MARGIN = 0.05
POSITION_RANK = 2
THREE_DIMENSIONS = 3


class AgentsVisual(BaseVisual):
    """Visual that logs the agents of a group as one batch per frame.

    The agents are colored by a state. Agents on a grid or in a continuous space of two dimensions are logged as
    `Points2D` at their cell centres or positions, agents in a continuous space of three dimensions as `Points3D` at
    their positions, and agents on a network as `GraphNodes` at their nodes, which keep the `pos` attributes of the
    nodes, or the positions of a seeded spring layout of the network. On a grid the radius is in cells, and on a network
    a fraction of the spacing of neighbouring nodes. Without a given radius, agents in a continuous space take a share
    of the side of the space, smaller for larger groups. Agents in a continuous space of two dimensions are logged with
    their second coordinate negated, so the view, bounded by the space, draws them with the vertical axis up. Every
    layer of regions of the space is logged once as static `LineStrips2D` under `<path>/regions/<layer>`, under the
    points of the agents. Every point carries the agent id as a hidden label, so agents keep their identity across
    frames and removed agents disappear. Agents whose position is a state are logged at that state: positions of one
    coordinate, such as cars on a road, as one thin series of one color per agent under `<path>/<position>`, so their
    time series view, without a legend, is a space-time diagram, and positions of two coordinates as `Points2D` with the
    vertical axis up.

    Args:
        color_by:
            The name of a numeric state that colors the agents, or `None` for one color.

        color_range:
            The values at the two ends of the color scale, or `None` for a range widened to every logged value.

        radius:
            The radius of the points, or `None` for 0.4 on grids and networks and a share of the side of other spaces.

        position:
            The name of a state that holds the position of every agent, or `None` for the location in the space.

        path:
            The entity path of the element or agent group to draw, which also names it in the problem.

        recording:
            The Rerun recording to log to, or `None` for the active recording of the thread.

    Attributes:
        styled_:
            The entity paths whose series style or static region outlines are logged.

        span_:
            The range of the colored values logged so far, or `None`.

    Examples:
        >>> from skcomplex.core.engine.viz import AgentsVisual
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
        path: str | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        self.color_by = color_by
        self.color_range = color_range
        self.radius = radius
        self.position = position
        self.path = path
        self.recording = recording

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Keep the agents, forget what earlier runs left, and set the views.

        What earlier runs left is the styled series and the color range.
        """
        self.styled_: set[str] = set()
        self.span_: tuple[float, float] | None = None
        super()._prepare(context)

    def _get_agent_colors(self, snapshot: AgentSnapshotSpec) -> npt.NDArray[np.uint8] | None:
        """Return one color per agent over the given range or a widened one.

        Without a given range, the range widens to every value so far.
        """
        if self.color_by is None or not len(snapshot.ids):
            return None
        values = np.asarray(snapshot.values[self.color_by], dtype=float)
        if self.color_range is not None:
            return BaseVisual._get_colors_of(values, *self.color_range)
        self.span_ = BaseVisual._get_widened(self.span_, values)
        return BaseVisual._get_colors_of(values, *self.span_)

    def _get_radius(self, space: BaseSpace | None, positions: npt.NDArray[np.float64]) -> float:
        """Return the radius of the agents.

        It is the given radius, a radius in cells on grids and networks, or a share of the side of other spaces.
        """
        if self.radius is not None:
            return float(self.radius)
        if space is None or isinstance(space, LatticeSpace | NetworkSpace):
            return GRID_RADIUS
        if isinstance(space, ContinuousSpace):
            return BaseVisual._get_radius_of(float(max(space.extent_)), len(positions))
        extent = float(np.ptp(positions, axis=0).max()) if len(positions) else 0.0
        return BaseVisual._get_radius_of(extent, len(positions))

    def _log_regions(self, recording: rr.RecordingStream, path: str, space: ContinuousSpace) -> None:
        """Log every layer of regions of a space once.

        They are logged as static outlines with the vertical axis up.
        """
        for layer, index in space.regions_.items():
            entity = f'{path}/regions/{layer}'
            if entity in self.styled_:
                continue
            rings = [ring for polygon in index.geometries for ring in (polygon.exterior, *polygon.interiors)]
            strips = [np.asarray(ring.coords) * UPWARD for ring in rings]
            recording.log(entity, rr.LineStrips2D(strips), static=True)
            self.styled_.add(entity)

    def _log_agents(
        self,
        recording: rr.RecordingStream,
        path: str,
        snapshot: AgentSnapshotSpec,
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
        colors, labels = self._get_agent_colors(snapshot), [str(agent) for agent in snapshot.ids]
        if isinstance(space, ContinuousSpace):
            self._log_regions(recording, path, space)
        if self.position is not None:
            positions = np.asarray(snapshot.values[self.position], dtype=float).reshape(len(snapshot.ids), -1)
            if positions.shape[1] == 1:
                entity = f'{path}/{self.position}'
                if entity not in self.styled_:
                    style = rr.SeriesLines(colors=[PATH_COLOR] * len(positions), widths=PATH_WIDTH)
                    recording.log(entity, style, static=True)
                    self.styled_.add(entity)
                recording.log(entity, rr.Scalars(positions[:, 0]))
                return
            positions = positions[:, :2] * UPWARD
            extent = float(np.ptp(positions, axis=0).max()) if len(positions) else 0.0
            radius = BaseVisual._get_radius_of(extent, len(positions)) if self.radius is None else float(self.radius)
            recording.log(path, rr.Points2D(positions, colors=colors, radii=radius, labels=labels, show_labels=False))
            return
        if snapshot.locations is None:
            return
        if isinstance(space, NetworkSpace):
            layout = self._get_layout(space)
            nodes = [str(node) for node in np.ravel(snapshot.locations).tolist()]
            recording.log(
                path,
                rr.GraphNodes(
                    nodes,
                    positions=[layout[node] for node in nodes],
                    colors=colors,
                    labels=labels,
                    show_labels=False,
                    radii=self._get_radius(space, np.zeros((0, 2))) * NODE_SPACING / 2,
                ),
            )
            return
        positions = np.asarray(snapshot.locations, dtype=float).reshape(len(snapshot.ids), -1)
        if isinstance(space, LatticeSpace):
            positions = positions + 0.5
        radius = self._get_radius(space, positions)
        if isinstance(space, ContinuousSpace) and space.n_dim_ == THREE_DIMENSIONS:
            recording.log(path, rr.Points3D(positions, colors=colors, radii=radius, labels=labels, show_labels=False))
            return
        if isinstance(space, ContinuousSpace):
            positions = positions * UPWARD
        recording.log(path, rr.Points2D(positions, colors=colors, radii=radius, labels=labels, show_labels=False))

    @staticmethod
    def _get_continuous_view(path: str, space: ContinuousSpace) -> rrb.View:
        """Return a 3D view of a box of three sides, or a 2D view of a box of two sides.

        The 2D view is bounded by the box.
        """
        if space.n_dim_ == THREE_DIMENSIONS:
            return rrb.Spatial3DView(origin=path, name=path)
        width, height = space.extent_[:2]
        margin = VIEW_MARGIN * max(width, height)
        bounds = rrb.VisualBounds2D(
            x_range=(-margin, width + margin),
            y_range=(-height - margin, margin),
        )
        return rrb.Spatial2DView(origin=path, name=path, visual_bounds=bounds)

    def _get_views(self, element: CompiledElementSpec | CompiledAgentsSpec, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return the view of the agents.

        It is a graph view of the whole network, or a 2D or 3D spatial view for agents in a space, or none without one.

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
            >>> from skcomplex.core.engine.runtime import simulate
            >>> from skcomplex.core.engine.viz import AgentsVisual
            >>> from skcomplex.systems.social.opinion_dynamics import VoterSystem
            >>> problem = simulate(VoterSystem(), until=0).problem
            >>> type(AgentsVisual(path=problem.agents[0].path).prepare(problem).views_[0]).__name__
            'GraphView'
        """
        if self.position is not None and isinstance(element, CompiledAgentsSpec):
            if np.ndim(np.asarray(element.values[self.position])) < POSITION_RANK:
                origin = f'{element.path}/{self.position}'
                return [rrb.TimeSeriesView(origin=origin, name=origin, plot_legend=rrb.PlotLegend(visible=False))]
            return [rrb.Spatial2DView(origin=element.path, name=element.path)]
        space = problem.agent_space
        if space is None:
            return []
        if isinstance(space, NetworkSpace):
            positions = np.array([[0.0, 0.0], *self._get_layout(space).values()])
            low, high = positions.min(axis=0) - NODE_SPACING, positions.max(axis=0) + NODE_SPACING
            bounds = rrb.VisualBounds2D(x_range=(low[0], high[0]), y_range=(low[1], high[1]))
            return [rrb.GraphView(origin=element.path, name=element.path, visual_bounds=bounds)]
        space = problem.agent_space
        if isinstance(space, ContinuousSpace):
            return [self._get_continuous_view(element.path, space)]
        return [rrb.Spatial2DView(origin=element.path, name=element.path)]
