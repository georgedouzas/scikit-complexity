"""Continuum percolation, the clusters of random overlapping disks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
import numpy.typing as npt

from ....core.components.elements import Element, GroupElement
from ....core.components.spaces import ContinuousSpace, StaticTime
from ....core.components.systems import BaseSystem, System
from ....core.specs import StateSpec, VisualSpec


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
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role="position")] = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x


class ContinuumPercolationSystem(BaseSystem):
    """Overlapping disks at uniform positions in a square.

    Their clusters percolate near a filling of 1.13.

    The number of disks is `filling side^2 / (pi radius^2)`. The clusters of touching disks are read by
    `clusters(source, radius=radius)`, and a `sweep` varies `filling` and `random_state` for the spanning
    probability (Satz 2022 ch. 3). The system has the positions `disks.x`.

    Args:
        filling:
            The filling factor, the total area of the disks over the area of the square.

        side:
            The side of the square.

        radius:
            The radius of a disk.

        random_state:
            The seed or generator of the positions.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ContinuumPercolationSystem
        >>> ContinuumPercolationSystem(1.0, side=10.0).prepare().system_.elements[0].size
        32
    """

    field = "physics/statistical_mechanics"
    topics = ("percolation", "disorder", "clusters", "static")
    references = ("Satz 2022", "Mertens and Moore 2012")

    def __init__(
        self,
        filling: float = 1.1,
        *,
        side: float = 60.0,
        radius: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.filling = filling
        self.side = side
        self.radius = radius
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        size = round(self.filling * self.side**2 / (np.pi * self.radius**2))
        positions = np.random.default_rng(self.random_state).uniform(0.0, self.side, (size, 2))
        visual = VisualSpec("points", {"position": "x", "radius": self.radius, "states": ()})
        disks = GroupElement("disks", element=DiskElement, size=size, values={"x": positions}, visual=visual)
        return System(
            self.label,
            elements=[disks],
            space=ContinuousSpace(extent=[self.side, self.side]),
            time=StaticTime(),
        )
