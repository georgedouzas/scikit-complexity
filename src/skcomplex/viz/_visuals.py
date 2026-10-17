"""Draw the state of elements as Rerun archetypes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import rerun as rr

from ..compile import CompiledElement
from ..core import Registry, VisualSpec
from ..runtime import Frame

MAX_POINT_DIMENSIONS = 3
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


class Visual:
    """Base class of visuals, which log the state of one element to a Rerun recording.

    Subclasses implement `log` and are registered under a name with `register_visual`.
    """

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
        >>> from skcomplex.viz import ScalarsVisual
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
        >>> from skcomplex.viz import PointsVisual
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
