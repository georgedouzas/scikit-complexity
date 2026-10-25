"""Draw lattice groups on group spaces as Cayley graphs, with sites on rings by word length colored by state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from ...core.base import BaseProblem
from ...core.problems import CompiledAgents, CompiledElement
from ...engine.runtime import Frame
from ...engine.viz import NODE_SPACING, UPWARD, BaseVisual
from ._spaces import CayleySpace

NODE_SHARE = 0.15


class CayleyVisual(BaseVisual):
    """Visual that logs every state of a lattice group on a group space as the nodes of its Cayley graph.

    Sites sit on rings by word length, with one ring per radius and the identity at the centre, colored by state.
    Every chosen state is logged under `<path>/<state>` as `GraphNodes` at every frame, with the edges from every site
    `g` to `g s` for every generator `s` logged once as static `GraphEdges`.

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

    Examples:
        >>> from skcomplex.formalisms.groups import CayleyVisual
        >>> CayleyVisual(low=0, high=1)
        CayleyVisual(high=1, low=0)
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
        super()._prepare(element, problem)

    def _log_edges(self, recording: rr.RecordingStream, path: str, space: CayleySpace) -> None:
        """Log the edges of every generator once, as static data."""
        edges = [(str(site), str(neighbour)) for site, neighbour, _ in space.edges_ if site != neighbour]
        recording.log(path, rr.GraphEdges(edges), static=True)
        self.edges_.add(path)

    def _log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log every chosen state of a lattice group on a group space as colored graph nodes.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled lattice group, with its path and the dotted names of its states.

            frame:
                The frame.
        """
        space = getattr(element.element, 'space', None)
        if not isinstance(space, CayleySpace):
            return
        positions = space.positions_ * NODE_SPACING * UPWARD
        nodes = [str(site) for site in range(space.size_)]
        for state, name in element.states.items():
            if self.states is not None and state not in self.states:
                continue
            path = f'{element.path}/{state}'
            if path not in self.edges_:
                self._log_edges(recording, path, space)
            values = np.ravel(np.asarray(frame.value(name), dtype=float))
            drawn = rr.GraphNodes(
                nodes,
                positions=positions,
                colors=self._colors(values, self.low, self.high),
                radii=NODE_SHARE * NODE_SPACING,
                show_labels=False,
            )
            recording.log(path, drawn)

    def _views(self, element: CompiledElement | CompiledAgents, problem: BaseProblem) -> list[rrb.View]:
        """Return one graph view per chosen state, bounded by the outer ring.

        Args:
            element:
                The compiled lattice group, with its path and the dotted names of its states.

            problem:
                The compiled problem the group belongs to.

        Returns:
            views:
                One graph view per chosen state, with `<path>/<state>` as origin.
        """
        space = getattr(getattr(element, 'element', None), 'space', None)
        if not isinstance(element, CompiledElement) or not isinstance(space, CayleySpace):
            return []
        extent = (float(space.lengths_.max(initial=0)) + 1.0) * NODE_SPACING
        bounds = rrb.VisualBounds2D(x_range=(-extent, extent), y_range=(-extent, extent))
        states = [state for state in element.states if self.states is None or state in self.states]
        return [rrb.GraphView(origin=f'{element.path}/{state}', name=state, visual_bounds=bounds) for state in states]
