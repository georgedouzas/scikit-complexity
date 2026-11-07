"""Continuum percolation, the clusters of random overlapping disks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
import numpy.typing as npt

from ....core.components.elements import Element, GroupElement
from ....core.components.spaces import ContinuousSpace, StaticTime
from ....core.components.system import System
from ....core.specs import StateSpec, VisualSpec
from ..._catalog import register_system


class DiskElement(Element):
    """Disk of the plane at a position that no law moves.

    Args:
        label:
            The label of the disk or the group.

        x:
            The position of its centre.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import DiskElement
        >>> DiskElement('disk', x=(1.0, 2.0)).x
        (1.0, 2.0)
    """

    def __init__(
        self,
        label: str,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x


@register_system(
    field='physics/statistical_mechanics',
    tags=('percolation', 'disorder', 'clusters', 'static'),
    references=('Satz 2022', 'Mertens and Moore 2012'),
)
def continuum_percolation(
    filling: float = 1.1,
    *,
    side: float = 60.0,
    radius: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return overlapping disks at uniform positions in a square, whose clusters percolate near a filling of 1.13.

    The number of disks is `filling side^2 / (pi radius^2)`. The clusters of touching disks are read by
    `clusters(source, radius=radius)`, and an `Experiment` sweeps `filling` and `random_state` for the spanning
    probability (Satz 2022 ch. 3).

    Args:
        filling:
            The filling factor, the total area of the disks over the area of the square.

        side:
            The side of the square.

        radius:
            The radius of a disk.

        random_state:
            The seed or generator of the positions.

    Returns:
        system:
            The system, with the positions `disks.x`.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import continuum_percolation
        >>> continuum_percolation(1.0, side=10.0).elements[0].size
        32
    """
    size = round(filling * side**2 / (np.pi * radius**2))
    positions = np.random.default_rng(random_state).uniform(0.0, side, (size, 2))
    visual = VisualSpec('points', {'position': 'x', 'radius': radius, 'states': ()})
    disks = GroupElement('disks', element=DiskElement, size=size, values={'x': positions}, visual=visual)
    return System(elements=[disks], space=ContinuousSpace(extent=[side, side]), time=StaticTime())
