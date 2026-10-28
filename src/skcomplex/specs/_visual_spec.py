"""Define the visual spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class VisualSpec(BaseSpec):
    """Name of a registered visual and the options it is built with.

    Args:
        kind:
            The name of a visual registered in the visualization layer.

        options:
            The keyword arguments of the visual.

    Examples:
        >>> from skcomplex.specs import VisualSpec
        >>> VisualSpec('points', {'position': 'x'}).kind
        'points'
    """

    kind: str = 'scalars'
    options: dict[str, object] = field(default_factory=dict)
