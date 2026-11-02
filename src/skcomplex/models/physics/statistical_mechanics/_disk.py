"""A disk at a fixed position of a continuous space."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt

from ....core.elements import Element
from ....specs import StateSpec


class DiskElement(Element):
    """Disk of the plane at a position that no law moves.

    Args:
        label:
            The label of the disk or the group.

        x:
            The position of its centre.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import DiskElement
        >>> DiskElement('disk', x=(1.0, 2.0)).x
        (1.0, 2.0)
    """

    def __init__(self, label: str, x: Annotated[npt.ArrayLike, StateSpec(vector=True, position=True)] = 0.0) -> None:
        super().__init__(label)
        self.x = x
