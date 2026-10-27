"""Define the base class of visuals and the registry that resolves them by name."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ...core.spaces import BaseSpace, NetworkSpace
from ..problems import BaseProblem, CompiledAgents, CompiledElement
from ..runtime import Frame
from ..solvers import AgentSnapshot
from ..utils import BaseEstimator
from ._colors import palette_colors
from ._utils import _network_layout


class BaseVisual(BaseEstimator):
    """Base class of visuals, which log the state of one element or agent group to a Rerun recording.

    `prepare(element, problem)` keeps the element, forgets what earlier frames left, and sets the Rerun views that show
    what the visual logs. `run(frame, recording)` logs the element at one frame: a compiled element through the hook
    `_log`, and agents through the hook `_log_agents` with the snapshot of the frame. Subclasses implement `_log`,
    `_log_agents`, or both, declare their views with `_views`, and are named by the kind of a `VisualSpec` in `VISUALS`.

    Attributes:
        element_:
            The compiled element or agents the visual logs.

        space_:
            The space the agents of the problem live in, or `None`.

        layouts_:
            The positions of the nodes of every network drawn so far, by the id of its graph.

        views_:
            The Rerun views that show what the visual logs, each with an entity path of the element as origin.
    """

    def _prepare(self, element: CompiledElement | CompiledAgents, problem: BaseProblem) -> None:
        """Keep the element and the space of the agents, forget earlier layouts, and set the views."""
        self.element_ = element
        self.space_: BaseSpace | None = problem.agent_space
        self.layouts_: dict[int, dict[str, npt.NDArray[np.float64]]] = {}
        self.views_ = self._views(element, problem)

    def _run(self, frame: Frame, recording: rr.RecordingStream) -> None:
        """Log the element at a frame, or the agents of the frame when the element is an agent group."""
        if isinstance(self.element_, CompiledAgents):
            snapshot = frame.agents.get(self.element_.path)
            if snapshot is not None:
                self._log_agents(recording, self.element_.path, snapshot, self.space_)
            return
        self._log(recording, self.element_, frame)

    def _views(self, element: CompiledElement | CompiledAgents, problem: BaseProblem) -> list[rrb.View]:
        """Return the Rerun views that show what the visual logs for an element, none by default."""
        return []

    def _log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log the state of a compiled element at one frame; a subclass that logs elements implements it."""
        raise NotImplementedError

    def _log_agents(
        self,
        recording: rr.RecordingStream,
        path: str,
        snapshot: AgentSnapshot,
        space: BaseSpace | None,
    ) -> None:
        """Log the agents of a group at one frame; log nothing by default."""

    def _colors(
        self,
        values: npt.NDArray[np.float64],
        low: float | None,
        high: float | None,
    ) -> npt.NDArray[np.uint8]:
        """Return the colors of values over the given range, or over the extremes of the values for a missing end."""
        low = float(values.min()) if low is None else float(low)
        high = float(values.max()) if high is None else float(high)
        return palette_colors(values, low, high)

    def _layout(self, space: NetworkSpace) -> dict[str, npt.NDArray[np.float64]]:
        """Return the positions of the nodes of a network, computed once per network."""
        if id(space.graph) not in self.layouts_:
            self.layouts_[id(space.graph)] = _network_layout(space.graph)
        return self.layouts_[id(space.graph)]
