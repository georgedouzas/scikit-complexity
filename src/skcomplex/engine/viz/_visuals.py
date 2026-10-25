"""Define the base class of visuals and the registry that resolves them by name."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import rerun as rr
import rerun.blueprint as rrb

from ...core.base import BaseProblem, Registry, VisualSpec
from ...core.problems import CompiledAgents, CompiledElement
from ..runtime import Frame


class BaseVisual:
    """Base class of visuals, which log the state of one element to a Rerun recording.

    Subclasses implement `log`, declare the Rerun views that show what they log with `views`, and are registered under a
    name with `register_visual`.
    """

    def views(self, element: CompiledElement | CompiledAgents, problem: BaseProblem) -> list[rrb.View]:
        """Return the Rerun views that show what the visual logs for an element, none by default.

        Args:
            element:
                The compiled element or agents, with their path.

            problem:
                The compiled problem the element belongs to.

        Returns:
            views:
                The views, each with an entity path of the element as origin.
        """
        return []

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


VISUALS = Registry[type[BaseVisual]]('visual')


def register_visual(kind: str, visual: type[BaseVisual]) -> None:
    """Register a visual class under the name element types refer to.

    Args:
        kind:
            The name used in `VisualSpec.kind`.

        visual:
            The visual class, built with the options of the `VisualSpec`.
    """
    VISUALS.register(kind, visual)


def resolve_visual(spec: VisualSpec) -> BaseVisual:
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
