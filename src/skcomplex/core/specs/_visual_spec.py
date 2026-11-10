"""Define the visual spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class VisualSpec(BaseSpec):
    """A spec of the visual of an element and the options it is built with.

    Args:
        kind:
            The name of a visual of the engine, such as `'points'`, or a visual class, such as one a system package
            defines on `BaseVisual`.

        options:
            The keyword arguments of the visual.

    Examples:
        >>> from skcomplex.core.specs import VisualSpec
        >>> VisualSpec('points', {'position': 'x'}).kind
        'points'
    """

    kind: str | type = 'scalars'
    options: dict[str, object] = field(default_factory=dict)
