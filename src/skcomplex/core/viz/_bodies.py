"""Draw bodies, the rods that join them, their fading trails, and static outlines from the states of an element."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ..base import ValidationError
from ..compile import CompiledAgents, CompiledElement, Problem
from ..runtime import Frame
from ._visuals import TRAIL_LENGTH, UPWARD, Visual, _log_scalars, _radius_of, register_visual

type Positions = Callable[..., npt.ArrayLike] | Sequence[str]

BODY_COLOR = (250, 180, 60)
ROD_COLOR = (220, 220, 220)
TRAIL_COLOR = (90, 170, 230)
OUTLINE_COLOR = (200, 200, 200)
MARK_COLOR = (150, 150, 150)
FAINTEST = 20
ROD_SHARE = 0.25
TRAIL_SHARE = 0.1
MARGIN = 0.1
PAIR = 2


def _values_of(element: CompiledElement, read: Callable[[str], Any]) -> dict[str, Any]:
    """Return the parameters and states of an element by their short names, read by dotted name."""
    names = {**element.parameters, **element.states}
    return {name: read(dotted) for name, dotted in names.items()}


def _fading_segments(
    paths: Sequence[npt.NDArray[np.float64]],
) -> tuple[list[npt.NDArray[np.float64]], list[tuple[int, int, int, int]]]:
    """Return the segments of paths and their colors, more transparent for older segments."""
    segments: list[npt.NDArray[np.float64]] = []
    colors: list[tuple[int, int, int, int]] = []
    for path in paths:
        count = len(path) - 1
        for index in range(count):
            segments.append(path[index : index + 2])
            alpha = round(FAINTEST + (255 - FAINTEST) * (index + 1) / count)
            colors.append((*TRAIL_COLOR, alpha))
    return segments, colors


def _bounds_of(points: npt.NDArray[np.float64]) -> rrb.VisualBounds2D | None:
    """Return 2D view bounds around points with a margin, in logged coordinates, or `None` for one point."""
    low, high = points.min(axis=0), points.max(axis=0)
    extent = float((high - low).max())
    if extent <= 0:
        return None
    center, half = (low + high) / 2, (1 + MARGIN) * extent / 2
    return rrb.VisualBounds2D(
        x_range=(center[0] - half, center[0] + half),
        y_range=(-(center[1] + half), -(center[1] - half)),
    )


class BodiesVisual(Visual):
    """Visual that draws bodies at positions computed from the states of an element, with rods, trails, and outlines.

    The bodies are logged as `Points2D` under `<path>/bodies`, the rods from the pivot through the bodies in order as
    one line strip under `<path>/rods`, the trails of the traced bodies as segments under `<path>/trail` that grow more
    transparent with age, the outlines once as static line strips under `<path>/outline`, and the marks once as static
    points under `<path>/marks`. Every state is also logged as a time series under `<path>/<state>`. The second
    coordinate is negated, since Rerun's 2D views point their vertical axis down, so the views draw it upward.

    Args:
        positions:
            A function of the parameters and states of the element by keyword that returns one row `(x, y)` per body,
            or the names of the two states that hold the position of a single body.

        pivot:
            The fixed point the rods start from, or `None` to start them at the first body.

        rods:
            Whether to draw rods through the bodies.

        traced:
            The indices of the bodies whose trails are drawn, or `None` for every body.

        trail:
            The number of recent positions in a trail, 0 for no trail, or `None` for the whole path.

        outline:
            Static polylines, each an array of rows `(x, y)`, such as the walls of a table.

        marks:
            Static points, such as fixed attracting centres.

        radius:
            The radius of the bodies, or `None` for a share of the extent of the view.

    Examples:
        >>> from skcomplex.core.viz import BodiesVisual
        >>> BodiesVisual(('x', 'y'), trail=None).trail is None
        True
    """

    def __init__(
        self,
        positions: Positions,
        *,
        pivot: Sequence[float] | None = None,
        rods: bool = False,
        traced: Sequence[int] | None = None,
        trail: int | None = TRAIL_LENGTH,
        outline: Sequence[npt.ArrayLike] = (),
        marks: Sequence[Sequence[float]] = (),
        radius: float | None = None,
    ) -> None:
        self.positions = positions
        self.pivot = pivot
        self.rods = rods
        self.traced = traced
        self.trail = trail
        self.outline = outline
        self.marks = marks
        self.radius = radius
        self._paths: deque[npt.NDArray[np.float64]] = deque(maxlen=trail if trail else None)
        self._drawn: set[str] = set()

    def _bodies(self, values: dict[str, Any]) -> npt.NDArray[np.float64]:
        """Return the positions of the bodies as rows `(x, y)`."""
        if callable(self.positions):
            points = np.asarray(self.positions(**values), dtype=float)
        else:
            points = np.array([[float(np.ravel(values[name])[0]) for name in self.positions]])
        points = np.atleast_2d(points)
        if points.ndim != PAIR or points.shape[1] != PAIR:
            error_msg = (
                f'The positions of the bodies visual have shape {points.shape}, but they must be pairs of coordinates, '
                'one row (x, y) per body.'
            )
            raise ValidationError(error_msg)
        return points

    def _chain(self, bodies: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the points the rods join, from the pivot when there is one."""
        return bodies if self.pivot is None else np.vstack([np.asarray(self.pivot, dtype=float), bodies])

    def _scale(self, bodies: npt.NDArray[np.float64]) -> float:
        """Return the extent of the drawing: the outline, the marks, the pivot, and the rods at full length."""
        chain = self._chain(bodies)
        pieces = [chain, *(np.atleast_2d(np.asarray(item, dtype=float)) for item in self.outline)]
        if self.marks:
            pieces.append(np.asarray(self.marks, dtype=float))
        points = np.vstack(pieces)
        extent = float(np.ptp(points, axis=0).max())
        if self.rods and len(chain) > 1:
            extent = max(extent, 2 * float(np.linalg.norm(np.diff(chain, axis=0), axis=1).sum()))
        return extent

    def _log_static(self, recording: rr.RecordingStream, path: str) -> None:
        """Log the outline and the marks once."""
        if path in self._drawn:
            return
        self._drawn.add(path)
        if self.outline:
            strips = [np.asarray(item, dtype=float) * UPWARD for item in self.outline]
            recording.log(f'{path}/outline', rr.LineStrips2D(strips, colors=OUTLINE_COLOR), static=True)
        if self.marks:
            marks = np.asarray(self.marks, dtype=float) * UPWARD
            recording.log(f'{path}/marks', rr.Points2D(marks, colors=MARK_COLOR), static=True)

    def _log_trail(
        self,
        recording: rr.RecordingStream,
        path: str,
        bodies: npt.NDArray[np.float64],
        width: float,
    ) -> None:
        """Log the fading trails of the traced bodies."""
        if self.trail == 0:
            return
        traced = list(range(len(bodies))) if self.traced is None else list(self.traced)
        self._paths.append(bodies[traced])
        if len(self._paths) < PAIR:
            return
        history = np.stack(self._paths)
        segments, colors = _fading_segments([history[:, column] for column in range(len(traced))])
        recording.log(f'{path}/trail', rr.LineStrips2D(segments, radii=width, colors=colors))

    def log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log the bodies, the rods, the trails, the outline, and the states of an element at one frame.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path and the dotted names of its variables.

            frame:
                The frame.

        Raises:
            ValidationError:
                The positions are not pairs of coordinates.
        """
        path = element.path
        bodies = self._bodies(_values_of(element, frame.value))
        share = _radius_of(self._scale(bodies), len(bodies))
        radius = share if self.radius is None else float(self.radius)
        self._log_static(recording, path)
        shown = bodies * UPWARD
        recording.log(f'{path}/bodies', rr.Points2D(shown, radii=radius, colors=BODY_COLOR))
        if self.rods:
            chain = self._chain(bodies) * UPWARD
            recording.log(f'{path}/rods', rr.LineStrips2D([chain], radii=share * ROD_SHARE, colors=ROD_COLOR))
        self._log_trail(recording, path, shown, share * TRAIL_SHARE)
        for state, name in element.states.items():
            _log_scalars(recording, f'{path}/{state}', frame.value(name))

    def views(self, element: CompiledElement | CompiledAgents, problem: Problem) -> list[rrb.View]:
        """Return a 2D view of the bodies that covers the drawing, and a time series view of the states.

        Args:
            element:
                The compiled element, with its path and the dotted names of its variables.

            problem:
                The compiled problem the element belongs to.

        Returns:
            views:
                A 2D spatial view with the element path as origin, bounded by the outline, the marks, the pivot, and
                the rods at full length, and a time series view of the states.
        """
        if not isinstance(element, CompiledElement):
            return []

        def read(name: str) -> Any:  # noqa: ANN401
            layout, vector = (
                (problem.states, problem.initial_state)
                if name in problem.states
                else (problem.parameters, problem.initial_parameters)
            )
            return layout.read(vector, name)

        bodies = self._bodies(_values_of(element, read))
        chain = self._chain(bodies)
        half = self._scale(bodies) / 2
        center = chain[0] if self.rods and self.pivot is not None else None
        points = np.vstack([center - half, center + half]) if center is not None else np.vstack([chain, bodies])
        if center is None and (self.outline or self.marks):
            pieces = [np.atleast_2d(np.asarray(item, dtype=float)) for item in self.outline]
            pieces += [np.asarray(self.marks, dtype=float)] if self.marks else []
            points = np.vstack([points, *pieces])
        path = element.path
        contents = [f'+ {path}/{state}/**' for state in element.states]
        return [
            rrb.Spatial2DView(origin=path, name=path, visual_bounds=_bounds_of(points)),
            rrb.TimeSeriesView(origin=path, contents=contents, name=f'{path} states'),
        ]


register_visual('bodies', BodiesVisual)
