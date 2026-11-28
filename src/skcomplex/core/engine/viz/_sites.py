"""Draw the values of a group on a triangular grid or a network as sites."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ...components.spaces import LatticeSpace, NetworkSpace
from ..problems import BaseProblemSpec, CompiledElementSpec, FrameSpec, NetworkSnapshotSpec
from ..problems._base_problem_spec import _CompiledAgents
from ._base import GRID_RADIUS, NODE_SPACING, BaseVisual

HEXAGON_RADIUS = float(1 / np.sqrt(3.0))
CHANGING_REACH = 10.0
CHANGING_LINK = 40.0
CHANGING_REPULSION = -60.0
CHANGING_CENTRING = 0.05


class SiteVisual(BaseVisual):
    """A visual that draws the values of a group on a triangular grid or a network.

    It colours each site by value, from `low` to `high`, such as `SiteVisual(low=0, high=2)` for three states of a
    cell. A triangular grid draws a hexagon around each site, and a network draws its nodes at their `pos` attributes or
    a fixed spring layout, with its edges drawn once. A network whose links change draws the nodes and edges present at
    each frame. It shows the infected people of a contact network, the cells of a hexagonal automaton, and the opinions
    of a growing social network.

    Read more in the user guide, under Observe.

    Args:
        states:
            The states to log, or `None` for every state.

        low:
            The value of the darkest color, or `None` for the smallest value of every frame.

        high:
            The value of the brightest color, or `None` for the largest value of every frame.

        path:
            The entity path of the element or agent group to draw, which also names it in the problem.

        recording:
            The Rerun recording to log to, or `None` for the active recording of the thread.

    Attributes:
        edges_:
            The entity paths whose static edges are logged.

        changing_:
            The hash of the latest logged edges of every changing network, by entity path.

    See Also:
        ImageVisual: Draws the values of a square grid as images.
        AgentsVisual: Draws agents at the nodes they occupy.

    Examples:
        >>> from skcomplex.core.engine.viz import SiteVisual
        >>> SiteVisual(low=0, high=2)
        SiteVisual(high=2, low=0)
    """

    numeric_options = ('low', 'high')

    def __init__(
        self,
        states: Sequence[str] | None = None,
        low: float | None = None,
        high: float | None = None,
        *,
        path: str | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        self.states = states
        self.low = low
        self.high = high
        self.path = path
        self.recording = recording

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Keep the group, forget the edges logged in earlier runs, and set the views."""
        self.edges_: set[str] = set()
        self.changing_: dict[str, int] = {}
        super()._prepare(context)

    def _log_graph(
        self,
        recording: rr.RecordingStream,
        path: str,
        space: NetworkSpace,
        values: npt.NDArray[np.float64],
    ) -> None:
        """Log the nodes of a network colored by value, and its edges once."""
        layout = self._get_layout(space)
        nodes = [str(node) for node in space.graph.nodes]
        if path not in self.edges_:
            edges = [(str(source), str(target)) for source, target, *_ in space.graph.edges]
            recording.log(path, rr.GraphEdges(edges), static=True)
            self.edges_.add(path)
        positions = [layout[node] for node in nodes]
        radius = GRID_RADIUS * NODE_SPACING / 2
        drawn = rr.GraphNodes(
            nodes,
            positions=positions,
            colors=self._get_colors(values, self.low, self.high),
            radii=radius,
            show_labels=False,
        )
        recording.log(path, drawn)

    def _log_changing(
        self,
        recording: rr.RecordingStream,
        path: str,
        snapshot: NetworkSnapshotSpec,
        values: npt.NDArray[np.float64],
    ) -> None:
        """Log the nodes present of a changing network colored by value.

        Its edges are logged when they change.
        """
        edges = [(str(source), str(target)) for source, target in zip(snapshot.sources, snapshot.targets, strict=True)]
        key = hash(tuple(edges))
        if self.changing_.get(path) != key:
            recording.log(path, rr.GraphEdges(edges, graph_type='undirected'))
            self.changing_[path] = key
        nodes = [str(node) for node in snapshot.nodes]
        colors = (
            self._get_colors(values[snapshot.nodes], self.low, self.high) if nodes else np.zeros((0, 4), dtype=np.uint8)
        )
        radius = GRID_RADIUS * NODE_SPACING / 2
        recording.log(path, rr.GraphNodes(nodes, colors=colors, radii=radius, show_labels=False))

    def _log(self, recording: rr.RecordingStream, element: CompiledElementSpec, frame: FrameSpec) -> None:
        """Log every chosen state of a triangular lattice or a network.

        They are logged under `<path>/<state>`.

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
            values = np.ravel(np.asarray(frame.get_value(name), dtype=float))
            path = BaseVisual._get_state_path(element.path, state)
            topology = frame.problem.topology
            if topology is not None and self._changes_topology(element, frame.problem):
                snapshot = topology.read_network(frame.parameters, element.path)
                self._log_changing(recording, path, snapshot, values)
                continue
            if isinstance(space, NetworkSpace):
                self._log_graph(recording, path, space, values)
                continue
            lattice = space if isinstance(space, LatticeSpace) else LatticeSpace(values.size).prepare()
            positions = lattice.positions_
            if positions.shape[1] == 1:
                positions = np.column_stack([positions[:, 0], np.zeros(len(positions))])
            radius = HEXAGON_RADIUS * min(lattice.spacings_)
            recording.log(
                path,
                rr.Points2D(positions, colors=self._get_colors(values, self.low, self.high), radii=radius),
            )

    def _get_views(
        self,
        element: CompiledElementSpec | _CompiledAgents,
        problem: BaseProblemSpec,
    ) -> list[rrb.View]:
        """Return the views of the chosen states of a group.

        A group on a network or a changing topology gets one graph view per state, and others 2D spatial views.

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
            >>> from skcomplex.core.components.elements import Element
            >>> from skcomplex.core.components.interactions import Interaction
            >>> from skcomplex.core.components.laws import RateLaw
            >>> from skcomplex.core.components.spaces import NetworkSpace
            >>> from skcomplex.core.systems import System
            >>> from skcomplex.core.engine.viz import SiteVisual
            >>> nodes = Element('nodes', values={'value': 1.0}, space=NetworkSpace(nx.cycle_graph(4)))
            >>> def fade(s):
            ...     return [RateLaw(s.target.value, -s.target.value)]
            >>> decay = Interaction('decay', roles={'target': 'nodes'}, laws=fade)
            >>> system = System(elements=[nodes], interactions=[decay])
            >>> problem = system.prepare().simulate(until=0).problem
            >>> views = SiteVisual(path=problem.elements[0].path).prepare(problem).views_
            >>> [type(view).__name__ for view in views]
            ['GraphView']
        """
        if not isinstance(element, CompiledElementSpec):
            return []
        space = getattr(element.element, 'space', None)
        states = [state for state in element.states if self.states is None or state in self.states]
        if self._changes_topology(element, problem):
            reach = CHANGING_REACH * float(np.sqrt(element.members or 1))
            bounds = rrb.VisualBounds2D(x_range=(-reach, reach), y_range=(-reach, reach))
            return [
                rrb.GraphView(
                    origin=BaseVisual._get_state_path(element.path, state),
                    name=state,
                    visual_bounds=bounds,
                    force_link=rrb.archetypes.ForceLink(distance=CHANGING_LINK),
                    force_many_body=rrb.archetypes.ForceManyBody(strength=CHANGING_REPULSION),
                    force_position=rrb.archetypes.ForcePosition(strength=CHANGING_CENTRING),
                )
                for state in states
            ]
        if not isinstance(space, NetworkSpace):
            return [
                rrb.Spatial2DView(origin=BaseVisual._get_state_path(element.path, state), name=state)
                for state in states
            ]
        positions = np.array(list(self._get_layout(space).values()))
        low, high = positions.min(axis=0) - NODE_SPACING, positions.max(axis=0) + NODE_SPACING
        bounds = rrb.VisualBounds2D(x_range=(low[0], high[0]), y_range=(low[1], high[1]))
        return [
            rrb.GraphView(origin=BaseVisual._get_state_path(element.path, state), name=state, visual_bounds=bounds)
            for state in states
        ]

    @staticmethod
    def _changes_topology(element: CompiledElementSpec, problem: BaseProblemSpec) -> bool:
        """Check whether a group lies on the nodes of a network whose topology changes."""
        topology = problem.topology
        return topology is not None and element.path in {path for path, _ in topology.layers}
