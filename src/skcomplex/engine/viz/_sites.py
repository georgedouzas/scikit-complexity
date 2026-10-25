"""Draw the states of a lattice group that is not square as colored sites or network nodes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ...core.base import BaseProblem
from ...core.problems import CompiledAgents, CompiledElement, NetworkSnapshot
from ...core.space import LatticeSpace, NetworkSpace
from ..runtime import Frame
from ._utils import GRID_RADIUS, NODE_SPACING
from ._visuals import BaseVisual, register_visual

HEXAGON_RADIUS = float(1 / np.sqrt(3.0))
CHANGING_REACH = 10.0
CHANGING_LINK = 40.0
CHANGING_REPULSION = -60.0
CHANGING_CENTRING = 0.05


def _changes_topology(element: CompiledElement, problem: BaseProblem) -> bool:
    """Check whether a group lies on the nodes of a network whose topology changes."""
    topology = getattr(problem, 'topology', None)
    return topology is not None and element.path in {path for path, _ in topology.layers}


class SiteVisual(BaseVisual):
    """Visual that logs every state of a lattice group that is not square as colored sites.

    A triangular lattice is logged as `Points2D` at the site positions, with radii that cover the hexagonal cells of
    the sites. A lattice group on a network is logged as `GraphNodes` colored by state, at the `pos` attributes of the
    nodes or a seeded spring layout, with its `GraphEdges` logged once as static data. A variable group on the nodes of
    a network whose topology changes logs the nodes present at every frame, placed by the force layout of the graph
    view, and its edges at every frame where they change.

    Args:
        states:
            The states to log, or `None` for every state.

        low:
            The value of the darkest color, or `None` for the smallest value of every frame.

        high:
            The value of the brightest color, or `None` for the largest value of every frame.

    Attributes:
        edges_:
            The entity paths whose static edges are logged.

        changing_:
            The hash of the latest logged edges of every changing network, by entity path.

    Examples:
        >>> from skcomplex.engine.viz import SiteVisual
        >>> SiteVisual(low=0, high=2)
        SiteVisual(high=2, low=0)
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

    def _prepare(self, element: CompiledElement | CompiledAgents, problem: BaseProblem) -> None:
        """Keep the group, forget the edges logged in earlier runs, and set the views."""
        self.edges_: set[str] = set()
        self.changing_: dict[str, int] = {}
        super()._prepare(element, problem)

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
        if path not in self.edges_:
            edges = [(str(source), str(target)) for source, target in space.graph.edges]
            recording.log(path, rr.GraphEdges(edges), static=True)
            self.edges_.add(path)
        positions = [layout[node] for node in nodes]
        radius = GRID_RADIUS * NODE_SPACING / 2
        drawn = rr.GraphNodes(
            nodes,
            positions=positions,
            colors=self._colors(values, self.low, self.high),
            radii=radius,
            show_labels=False,
        )
        recording.log(path, drawn)

    def _log_changing(
        self,
        recording: rr.RecordingStream,
        path: str,
        snapshot: NetworkSnapshot,
        values: npt.NDArray[np.float64],
    ) -> None:
        """Log the nodes present of a changing network colored by value, and its edges when they change."""
        edges = [(str(source), str(target)) for source, target in zip(snapshot.sources, snapshot.targets, strict=True)]
        key = hash(tuple(edges))
        if self.changing_.get(path) != key:
            recording.log(path, rr.GraphEdges(edges, graph_type='undirected'))
            self.changing_[path] = key
        nodes = [str(node) for node in snapshot.nodes]
        colors = (
            self._colors(values[snapshot.nodes], self.low, self.high) if nodes else np.zeros((0, 4), dtype=np.uint8)
        )
        radius = GRID_RADIUS * NODE_SPACING / 2
        recording.log(path, rr.GraphNodes(nodes, colors=colors, radii=radius, show_labels=False))

    def _log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
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
            if _changes_topology(element, frame.problem):
                snapshot = getattr(frame.problem, 'topology', None).read_network(frame.parameters, element.path)  # type: ignore[union-attr]
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
            recording.log(path, rr.Points2D(positions, colors=self._colors(values, self.low, self.high), radii=radius))

    def _views(self, element: CompiledElement | CompiledAgents, problem: BaseProblem) -> list[rrb.View]:
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
            >>> from skcomplex.engine.compile import compile_system
            >>> from skcomplex.core.components import ElementComponent
            >>> from skcomplex.core.space import NetworkSpace
            >>> from skcomplex.core.system import System
            >>> from skcomplex.engine.viz import SiteVisual
            >>> NodeComponent = ElementComponent.define('NodeComponent', states={'value': state(0.0)})
            >>> system = System(components=[NodeComponent.lattice('nodes', NetworkSpace(nx.cycle_graph(4)))])
            >>> problem = compile_system(system)
            >>> [type(view).__name__ for view in SiteVisual().prepare(problem.elements[0], problem).views_]
            ['GraphView']
        """
        if not isinstance(element, CompiledElement):
            return []
        space = getattr(element.element, 'space', None)
        states = [state for state in element.states if self.states is None or state in self.states]
        if not isinstance(space, NetworkSpace):
            return [rrb.Spatial2DView(origin=f'{element.path}/{state}', name=state) for state in states]
        if _changes_topology(element, problem):
            reach = CHANGING_REACH * float(np.sqrt(element.members or 1))
            bounds = rrb.VisualBounds2D(x_range=(-reach, reach), y_range=(-reach, reach))
            return [
                rrb.GraphView(
                    origin=f'{element.path}/{state}',
                    name=state,
                    visual_bounds=bounds,
                    force_link=rrb.archetypes.ForceLink(distance=CHANGING_LINK),
                    force_many_body=rrb.archetypes.ForceManyBody(strength=CHANGING_REPULSION),
                    force_position=rrb.archetypes.ForcePosition(strength=CHANGING_CENTRING),
                )
                for state in states
            ]
        positions = np.array(list(self._layout(space).values()))
        low, high = positions.min(axis=0) - NODE_SPACING, positions.max(axis=0) + NODE_SPACING
        bounds = rrb.VisualBounds2D(x_range=(low[0], high[0]), y_range=(low[1], high[1]))
        return [rrb.GraphView(origin=f'{element.path}/{state}', name=state, visual_bounds=bounds) for state in states]


register_visual('sites', SiteVisual)
