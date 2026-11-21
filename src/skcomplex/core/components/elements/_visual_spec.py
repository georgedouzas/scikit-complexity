"""Define the visual spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

from ...base import BaseSpec


@dataclass(frozen=True)
class VisualSpec(BaseSpec):
    """A spec of how an element is drawn, a visual and the options it is built with.

    It names a visual of the engine, such as `'points'` for a cloud of particles or `'line'` for a value over time, and
    the keyword arguments it takes, such as `{'position': 'x'}`. An element sets it as its class attribute `visual`, and
    a visualisation observer may draw the element otherwise. It draws the size of a population as a line, the birds of a
    flock as points, and a grid of cells as an image.

    Read more in the user guide, under Observe.

    Args:
        kind:
            The name of a visual of the engine, such as `'points'`, or a visual class, such as one a system package
            defines on `BaseVisual`.

        options:
            The keyword arguments of the visual.

    See Also:
        BaseElement: Sets the visual of its class.

    Examples:
        >>> from skcomplex.core.components.elements import VisualSpec
        >>> VisualSpec('points', {'position': 'x'}).kind
        'points'
    """

    kind: str | type = 'scalars'
    options: dict[str, object] = field(default_factory=dict)
