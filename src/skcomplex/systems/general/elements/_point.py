"""Define the element of a point that moves in a continuous space."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class PointElement(Element):
    """An element of a point that moves in a continuous space by its acceleration.

    The point has a vector state `x` of order 2, its position. The laws that act on it give the second derivative of
    `x`, and the companion state `x_d1` holds its velocity. The boundary of the space acts on `x`.

    Unlike `Element`, which declares its own states, it fixes one position of order 2 with its velocity. Choose it when
    members are bodies that move by forces in a continuous space.

    It appears in many fields. Atoms of a gas, stars of a cluster, birds of a flock, and pedestrians of a crowd are
    points that move by the forces on them.

    Args:
        label:
            The label of the point.

        x:
            The initial value of the vector state. A number applies to every component.

        x_d1:
            The initial value of the first derivative of the vector state. A number applies to every component.

    Attributes:
        default_visual_:
            Points at the values of `x`.

    Examples:
        >>> from skcomplex.systems.general.elements import PointElement
        >>> from skcomplex.core.components.spaces import ContinuousSpace
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> system = CompositeSystem(elements=[PointElement('ball', x_d1=[1.0, 0.0])], space=ContinuousSpace(n_dim=2))
        >>> simulate(system, until=2.0, dt=1.0).get_values('ball.x')[-1].tolist()
        [2.0, 0.0]
    """

    visual = VisualSpec('points', {'position': 'x'})

    def __init__(
        self,
        label: str,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, order=2, role='position')] = 0.0,
        x_d1: npt.ArrayLike = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.x_d1 = x_d1
