"""Define the element of a point that moves in a continuous space."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

import numpy.typing as npt

from ....core.components.elements import Element, VisualSpec
from ....core.components.spaces import BaseSpace


class PointElement(Element):
    """An element of a point at the position `x` that the forces on it move.

    The forces give the acceleration `x''` of the position, and the velocity is the value `x'`, which starts at zero
    unless `values` gives it.

    Unlike `Element`, which takes any values, it names the position `x` and draws it as a point. Choose it when members
    are bodies that move by forces in a continuous space.

    It appears in many fields. Atoms of a gas, stars of a cluster, birds of a flock, and pedestrians of a crowd are
    points that move by the forces on them.

    Args:
        label:
            The label of the point.

        x:
            The initial position. A number applies to every component.

        values:
            Further values by name, such as the starting velocity `"x'"`.

        size:
            The number of points, or `None` for one.

        space:
            The space whose sites hold one point each, or `None`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.general.elements import PointElement
        >>> from skcomplex.core.components.interactions import Interaction
        >>> from skcomplex.core.components.laws import RateLaw
        >>> ball = PointElement('ball', x=np.array([1.0, 0.0]), values={"x'": np.array([0.0, 1.0])})
        >>> spring = lambda s: [RateLaw(s.target.x.diff(s.t, 2), -s.target.x)]
        >>> well = Interaction('well', roles={'target': 'ball'}, laws=spring)
        >>> system = System(elements=[ball], interactions=[well]).prepare()
        >>> position = system.simulate(until=np.pi / 2, dt=np.pi / 2).get_values('ball.x')[-1]
        >>> round(float(position[1]), 3)
        1.0
    """

    visual = VisualSpec('points', {'position': 'x'})

    def __init__(
        self,
        label: str,
        x: npt.ArrayLike = 0.0,
        *,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.x = x
