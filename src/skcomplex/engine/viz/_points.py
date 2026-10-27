"""Draw a vector state of an element as points with trails."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import deque
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ..problems import BaseProblem, CompiledAgents, CompiledElement
from ..runtime import Frame
from ._colors import _widened, colors_of
from ._utils import UPWARD, _radius_of, log_scalars
from ._visuals import BaseVisual

MAX_POINT_DIMENSIONS = 3
TRAIL_LENGTH = 256
MAX_TRAIL_POINTS = 10
TRAIL_WIDTH = 0.25


class PointsVisual(BaseVisual):
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

    Attributes:
        history_:
            The recent positions of the points, the oldest first.

        bounds_:
            The lowest and highest coordinates of the positions logged so far, or `None`.

        span_:
            The range of the colored values logged so far, or `None`.

    Examples:
        >>> from skcomplex.engine.viz import PointsVisual
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

    def _prepare(self, element: CompiledElement | CompiledAgents, problem: BaseProblem) -> None:
        """Keep the element, forget the trail, the bounds, and the color range of earlier runs, and set the views."""
        self.history_: deque[npt.NDArray[np.float64]] = deque(maxlen=max(self.trail, 1))
        self.bounds_: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None
        self.span_: tuple[float, float] | None = None
        super()._prepare(element, problem)

    def _point_colors(self, element: CompiledElement, frame: Frame, count: int) -> npt.NDArray[np.uint8] | None:
        """Return the colors of the points over the given range, or over a range widened to every value so far."""
        if self.color_by is None:
            return None
        values = np.asarray(frame.value(element.states[self.color_by]), dtype=float).reshape(count, -1)
        magnitudes = np.linalg.norm(values, axis=1)
        if self.color_range is not None:
            return colors_of(magnitudes, *self.color_range)
        self.span_ = _widened(self.span_, magnitudes)
        return colors_of(magnitudes, *self.span_)

    def _extent_radius(self, points: npt.NDArray[np.float64]) -> float:
        """Return a share of the extent of every position logged so far, keeping the bounds of the positions."""
        low, high = points.min(axis=0), points.max(axis=0)
        if self.bounds_ is not None:
            low, high = np.minimum(low, self.bounds_[0]), np.maximum(high, self.bounds_[1])
        self.bounds_ = (low, high)
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
        if self.history_ and self.history_[-1].shape != points.shape:
            self.history_.clear()
        self.history_.append(points)
        strips = list(np.stack(self.history_, axis=1))
        radius = self._extent_radius(points) * TRAIL_WIDTH
        if points.shape[1] == MAX_POINT_DIMENSIONS:
            recording.log(f'{path}/trail', rr.LineStrips3D(strips, radii=radius, colors=colors))
        else:
            recording.log(f'{path}/trail', rr.LineStrips2D(strips, radii=radius, colors=colors))

    def _log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
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
        colors = self._point_colors(element, frame, len(points))
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
            log_scalars(recording, f'{element.path}/{state}', frame.value(element.states[state]))

    def _views(self, element: CompiledElement | CompiledAgents, problem: BaseProblem) -> list[rrb.View]:
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
            >>> from skcomplex.core.elements import GroupElement, ParticleElement
            >>> from skcomplex.core.spaces import ContinuousSpace
            >>> from skcomplex.core.system import System
            >>> from skcomplex.engine.compile import compile_system
            >>> from skcomplex.engine.viz import PointsVisual
            >>> gas = GroupElement('gas', element=ParticleElement, size=10, values={'x': [[0.0, 0.0, 0.0]] * 10})
            >>> problem = compile_system(System(elements=[gas], space=ContinuousSpace(n_dim=3)))
            >>> views = PointsVisual(position='x', states=()).prepare(problem.elements[0], problem).views_
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
