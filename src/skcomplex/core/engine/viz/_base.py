"""Define the base class of visuals.

A visual logs one element or agent group to a Rerun recording.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Self

import networkx as nx
import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ...base import BaseConfig
from ...components.spaces import BaseSpace, NetworkSpace
from ...components.specs import CompiledAgentsSpec, CompiledElementSpec
from ...errors import ValidationError
from ...specs import AgentSnapshotSpec
from ..problems import BaseProblemSpec, FrameSpec

_UPWARD = np.array([1.0, -1.0])
_LAYOUT_SEED = 0
_NODE_SPACING = 40.0
_POINT_SHARE = 0.02
_CROWD_SIZE = 10
_CROWD_EXPONENT = 0.25
_GRID_RADIUS = 0.4
_MODULUS = 'modulus'
_PHASE = 'phase'
PARTS = ('real', 'imag')
_SERIES_COLORS = (
    (230, 159, 0),
    (86, 180, 233),
    (0, 158, 115),
    (240, 228, 66),
    (0, 114, 178),
    (213, 94, 0),
    (204, 121, 167),
    (170, 170, 170),
)
_POINT_FLOOR = 0.25
_COLOR_SCALE = np.array(
    [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
    dtype=float,
)
_PALETTE = (
    np.stack(
        [
            np.interp(np.linspace(0.0, 1.0, 256), np.linspace(0.0, 1.0, len(_COLOR_SCALE)), _COLOR_SCALE[:, channel])
            for channel in range(3)
        ],
        axis=-1,
    )
    .round()
    .astype(np.uint8)
)
_PHASE_SCALE = np.array(
    [[230, 90, 80], [225, 190, 70], [90, 190, 110], [70, 160, 210], [150, 100, 200], [230, 90, 80]],
    dtype=float,
)
_PHASE_PALETTE = (
    np.stack(
        [
            np.interp(np.linspace(0.0, 1.0, 256), np.linspace(0.0, 1.0, len(_PHASE_SCALE)), _PHASE_SCALE[:, channel])
            for channel in range(3)
        ],
        axis=-1,
    )
    .round()
    .astype(np.uint8)
)


class BaseVisual(BaseConfig[BaseProblemSpec, FrameSpec]):
    """Base class of visuals.

    A visual logs the state of one element or agent group to a Rerun recording. A visual follows the common API of the
    engine. `prepare(problem)` finds the element or agent group of the problem at the entity path `path`, forgets what
    earlier frames left, and sets the Rerun views that show what the visual logs. `run(frame)` logs the element at one
    frame to the recording `recording`: a compiled element through the hook `_log`, and agents through the hook
    `_log_agents` with the snapshot of the frame. Subclasses implement `_log`, `_log_agents`, or both, declare their
    views with `_views`, and are named by the kind of a `VisualSpec`. `RerunObserver` builds the visual of every element
    and agent group with their path and its recording.

    Args:
        path:
            The entity path of the element or agent group to draw, which also names it in the problem.

        recording:
            The Rerun recording to log to, or `None` for the active recording of the thread.

    Attributes:
        element_:
            The compiled element or agents the visual logs.

        recording_:
            The Rerun recording the visual logs to, or `None` while no recording is active.

        space_:
            The space the agents of the problem live in, or `None`.

        layouts_:
            The positions of the nodes of every network drawn so far, by the id of its graph.

        views_:
            The Rerun views that show what the visual logs, each with an entity path of the element as origin.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.engine.viz import ScalarsVisual
        >>> from skcomplex.systems.biology.ecology import LotkaVolterraSystem
        >>> problem = simulate(LotkaVolterraSystem(), until=0).problem
        >>> visual = ScalarsVisual(path='prey').prepare(problem)
        >>> visual.element_.path, str(visual.views_[0].origin)
        ('prey', 'prey')
    """

    def __init__(self, *, path: str | None = None, recording: rr.RecordingStream | None = None) -> None:
        self.path = path
        self.recording = recording

    def _check_path(self) -> None:
        """Raise an error unless the path is a string.

        Raises:
            ValidationError:
                The path is not a string.
        """
        if not isinstance(self.path, str):
            error_msg = f'path = {self.path!r} must be the entity path of the element or agent group to draw.'
            raise ValidationError(error_msg)

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Find the element at the path, keep the recording and space, and set the views.

        It first raises an error unless the path is a string. The space kept is that of the agents.

        Args:
            context:
                The compiled problem.

        Raises:
            ValidationError:
                No problem is passed, or no element or agent group of the problem has the path of the visual.
        """
        self._check_path()
        if context is None:
            error_msg = 'A visual prepares on a compiled problem. Pass the problem to `prepare`.'
            raise ValidationError(error_msg)
        problem = context
        items: list[CompiledElementSpec | CompiledAgentsSpec] = [*problem.elements, *problem.agents]
        elements = {item.path: item for item in items}
        if self.path not in elements:
            error_msg = (
                f'No element or agent group of the problem has the path {self.path!r}. The paths are: '
                f'{", ".join(elements)}.'
            )
            raise ValidationError(error_msg)
        self.element_: CompiledElementSpec | CompiledAgentsSpec = elements[self.path]
        self.recording_ = self.recording if self.recording is not None else rr.get_data_recording()
        self.space_: BaseSpace | None = problem.agent_space
        self.layouts_: dict[int, dict[str, npt.NDArray[np.float64]]] = {}
        self.views_ = self._views(self.element_, problem)

    def _run(self, inputs: FrameSpec | None) -> None:
        """Log the element at a frame.

        When the element is an agent group, it logs the agents of the frame.

        Args:
            inputs:
                The frame.

        Raises:
            ValidationError:
                No frame is passed, or no Rerun recording is active.
        """
        if inputs is None:
            error_msg = 'A visual runs on a frame. Pass the frame to `run`.'
            raise ValidationError(error_msg)
        if self.recording_ is None:
            error_msg = 'No Rerun recording is active. Pass `recording` or start one with `rerun.init`.'
            raise ValidationError(error_msg)
        if isinstance(self.element_, CompiledAgentsSpec):
            snapshot = inputs.agents.get(self.element_.path)
            if snapshot is not None:
                self._log_agents(self.recording_, self.element_.path, snapshot, self.space_)
            return
        self._log(self.recording_, self.element_, inputs)

    def prepare(self, problem: BaseProblemSpec | None = None) -> Self:
        """Prepare the checked visual for the element at its path in a compiled problem.

        Args:
            problem:
                The compiled problem.

        Returns:
            self:
                The prepared visual, with its views `views_`.

        Raises:
            ValidationError:
                No element or agent group of the problem has the path of the visual.
        """
        return super().prepare(problem)

    def run(self, frame: FrameSpec | None = None, final: bool = False) -> Self:
        """Log the element or the agents of the visual at a frame.

        Args:
            frame:
                The frame.

            final:
                Whether this is the last call of the run.

        Returns:
            self:
                The visual.
        """
        super().run(frame, final)
        return self

    def _views(self, element: CompiledElementSpec | CompiledAgentsSpec, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return the Rerun views that show what the visual logs for an element.

        By default there are none.
        """
        return []

    def _log(self, recording: rr.RecordingStream, element: CompiledElementSpec, frame: FrameSpec) -> None:
        """Log the state of a compiled element at one frame.

        A subclass that logs elements implements it.
        """
        raise NotImplementedError

    def _log_agents(
        self,
        recording: rr.RecordingStream,
        path: str,
        snapshot: AgentSnapshotSpec,
        space: BaseSpace | None,
    ) -> None:
        """Log the agents of a group at one frame.

        By default, log nothing.
        """

    def _colors(
        self,
        values: npt.NDArray[np.float64],
        low: float | None,
        high: float | None,
    ) -> npt.NDArray[np.uint8]:
        """Return the colors of values over the given range.

        A missing end of the range is the extreme of the values.
        """
        low = float(values.min()) if low is None else float(low)
        high = float(values.max()) if high is None else float(high)
        return self._palette_colors(values, low, high)

    def _layout(self, space: NetworkSpace) -> dict[str, npt.NDArray[np.float64]]:
        """Return the positions of the nodes of a network, computed once per network."""
        if id(space.graph) not in self.layouts_:
            self.layouts_[id(space.graph)] = self._network_layout(space.graph)
        return self.layouts_[id(space.graph)]

    @staticmethod
    def _phase_colors(phases: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
        """Return the colors of phases in `(-pi, pi]` from a cyclic palette.

        `-pi` and `pi` thus share one color.
        """
        levels = np.mod(phases + np.pi, 2 * np.pi) * (255.0 / (2 * np.pi))
        return _PHASE_PALETTE[np.clip(levels, 0.0, 255.0).astype(np.uint8)]

    @staticmethod
    def _widened(span: tuple[float, float] | None, values: npt.NDArray[np.float64]) -> tuple[float, float]:
        """Return a color range widened to cover values.

        Without a range yet, it starts from their extremes.
        """
        low, high = float(np.min(values)), float(np.max(values))
        return (low, high) if span is None else (min(span[0], low), max(span[1], high))

    @staticmethod
    def _colors_of(values: npt.NDArray[np.float64], low: float, high: float) -> npt.NDArray[np.uint8]:
        """Return one color per value on a scale from blue at `low` to yellow at `high`.

        The colors are bright enough to see on black.

        Args:
            values:
                The values.

            low:
                The value of the darkest color.

            high:
                The value of the brightest color.

        Returns:
            colors:
                One RGB color per value.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.engine.viz import BaseVisual
            >>> BaseVisual._colors_of(np.array([0.0, 1.0]), 0.0, 1.0).shape
            (2, 3)
        """
        fractions = np.clip((values - low) / (high - low), 0.0, 1.0) if high > low else np.zeros_like(values)
        fractions = _POINT_FLOOR + (1.0 - _POINT_FLOOR) * fractions
        anchors = np.linspace(0.0, 1.0, len(_COLOR_SCALE))
        channels = [np.interp(fractions, anchors, _COLOR_SCALE[:, channel]) for channel in range(3)]
        return np.stack(channels, axis=-1).round().astype(np.uint8)

    @staticmethod
    def _palette_colors(values: npt.NDArray[np.float64], low: float, high: float) -> npt.NDArray[np.uint8]:
        """Return the colors of an array of values from a palette of 256 levels.

        It is fast enough for large images.

        Args:
            values:
                The values.

            low:
                The value of the darkest color.

            high:
                The value of the brightest color.

        Returns:
            colors:
                One RGB color per value, the darkest one for every value when the range is empty.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.engine.viz import BaseVisual
            >>> BaseVisual._palette_colors(np.array([0.0, 1.0]), 0.0, 1.0).shape
            (2, 3)
        """
        if high <= low:
            return np.broadcast_to(_PALETTE[0], (*values.shape, 3))
        levels = np.clip((values - low) * (255.0 / (high - low)), 0.0, 255.0).astype(np.uint8)
        return _PALETTE[levels]

    @staticmethod
    def _network_layout(graph: nx.Graph) -> dict[str, npt.NDArray[np.float64]]:
        """Return fixed positions of the nodes of a network.

        Neighbours lie about a spacing apart. The positions are the `pos` attributes of the nodes when every node has
        one, and a seeded spring layout otherwise.
        """
        scale = _NODE_SPACING * float(np.sqrt(max(graph.number_of_nodes(), 1))) / 2
        given = nx.get_node_attributes(graph, 'pos')
        if given and len(given) == graph.number_of_nodes():
            layout = nx.rescale_layout_dict(
                {node: np.asarray(position, dtype=float) for node, position in given.items()},
            )
        else:
            layout = nx.spring_layout(graph, seed=_LAYOUT_SEED)
        return {str(node): np.asarray(position, dtype=float) * scale for node, position in layout.items()}

    @staticmethod
    def _radius_of(extent: float, count: int) -> float:
        """Return a point radius that is a share of the extent of a view.

        It is smaller for crowds so they do not overlap.
        """
        return _POINT_SHARE * (extent if extent > 0 else 1.0) / max(count / _CROWD_SIZE, 1.0) ** _CROWD_EXPONENT

    @staticmethod
    def _log_scalars(recording: rr.RecordingStream, path: str, value: object) -> None:
        """Log the components of a value as scalar series.

        The real and imaginary parts of a complex value are logged apart.

        Args:
            recording:
                The Rerun recording.

            path:
                The entity path of the series.

            value:
                A number or an array of numbers, real or complex.

        Examples:
            >>> import rerun as rr
            >>> from skcomplex.core.engine.viz import BaseVisual
            >>> BaseVisual._log_scalars(rr.RecordingStream('example'), 'value', 1.0 + 2.0j)
        """
        values = np.ravel(np.asarray(value))
        if not np.iscomplexobj(values):
            recording.log(path, rr.Scalars(values))
            return
        for part, components in zip(PARTS, (values.real, values.imag), strict=True):
            recording.log(f'{path}/{part}', rr.Scalars(components))
