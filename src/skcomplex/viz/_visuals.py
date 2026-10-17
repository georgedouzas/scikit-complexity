"""Draw the state of elements as Rerun archetypes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np
import rerun as rr

from ..compile import CompiledElement
from ..core import Registry, VisualSpec
from ..runtime import Frame

MAX_POINT_DIMENSIONS = 3


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
            recording.log(f'{element.path}/{state}', rr.Scalars(np.atleast_1d(frame.value(name))))


class PointsVisual(Visual):
    """Visual that logs a vector state of an element as a point and its other states as time series.

    Args:
        position:
            The name of the vector state that holds the position.

        radius:
            The radius of the point.

    Examples:
        >>> from skcomplex.viz import PointsVisual
        >>> PointsVisual(position='x').position
        'x'
    """

    def __init__(self, position: str, radius: float = 0.05) -> None:
        self.position = position
        self.radius = radius

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
        position = np.atleast_1d(frame.value(element.states[self.position]))
        if position.size == MAX_POINT_DIMENSIONS:
            recording.log(element.path, rr.Points3D([position], radii=self.radius))
        elif position.size == MAX_POINT_DIMENSIONS - 1:
            recording.log(element.path, rr.Points2D([position], radii=self.radius))
        else:
            recording.log(f'{element.path}/{self.position}', rr.Scalars(position))
        for state, name in element.states.items():
            if state != self.position:
                recording.log(f'{element.path}/{state}', rr.Scalars(np.atleast_1d(frame.value(name))))


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
