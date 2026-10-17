"""Declare how the state of an element type is drawn."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VisualSpec:
    """Name of a registered visual and the options it is built with.

    Args:
        kind:
            The name of a visual registered in the visualization layer.

        options:
            The keyword arguments of the visual.

    Examples:
        >>> from skcomplex.core import VisualSpec
        >>> VisualSpec('points', {'position': 'x'}).kind
        'points'
    """

    kind: str = 'scalars'
    options: dict[str, Any] = field(default_factory=dict)
