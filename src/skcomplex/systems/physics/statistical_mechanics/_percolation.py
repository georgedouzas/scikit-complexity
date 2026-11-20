"""Continuum percolation, the clusters of random overlapping disks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System


class DiskElement(Element):
    """Disk of the plane at a position that no law moves.

    Args:
        label:
            The label of the disk or the group.

        x:
            The position of its centre.

        radius:
            The radius of the disk, which it is drawn with.

        size:
            The number of disks, or `None` for one.

        space:
            The plane of the disks, or `None` for none.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import DiskElement
        >>> DiskElement('disk', x=(1.0, 2.0)).x
        (1.0, 2.0)
    """

    visual = VisualSpec('points', {'position': 'x', 'radius': 'radius', 'states': ()})

    def __init__(
        self,
        label: str,
        x: npt.ArrayLike = 0.0,
        radius: Annotated[float, RealSet.open(0, oo)] = 1.0,
        *,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x
        self.radius = radius


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ContinuumPercolationSystem
        >>> ContinuumPercolationSystem(1.0, side=10.0).prepare().system_.elements[0].size
        32
    """

    field = 'physics/statistical_mechanics'
    topics = ('percolation', 'disorder', 'clusters', 'static')
    references = ('Satz 2022', 'Mertens and Moore 2012')

    def __init__(
        self,
        filling: float = 1.1,
        *,
        side: float = 60.0,
        radius: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.filling = filling
        self.side = side
        self.radius = radius
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the disks at uniform positions in the square.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        size = round(self.filling * self.side**2 / (np.pi * self.radius**2))
        positions = np.random.default_rng(self.random_state).uniform(0.0, self.side, (size, 2))
        square = ContinuousSpace(extent=[self.side, self.side])
        disks = DiskElement('disks', x=positions, radius=self.radius, size=size, space=square)
        self.system_ = System(self.label, elements=[disks], space=square)
