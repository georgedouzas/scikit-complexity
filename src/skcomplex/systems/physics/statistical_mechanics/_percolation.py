"""Continuum percolation, the clusters of random overlapping disks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import ContinuousSpace, StaticTime
from ....core.components.system import System
from ....core.specs import VisualSpec
from ..._catalog import register_system
from ._disk import DiskElement


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
    `ClusterAnalysis(radius=radius)`, and an `Experiment` sweeps `filling` and `random_state` for the spanning
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
