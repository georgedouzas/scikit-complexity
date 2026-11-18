"""Car on a road at the position `x`, whose speed is the derivative `x'`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

from .....core.components.elements import Element
from .....core.components.spaces import BaseSpace


class CarElement(Element):
    """Car on a road at the position `x`, whose speed is the derivative `x'`.

    The interactions that drive the car give the acceleration `x''`, and the speed `x'` starts at zero unless `values`
    gives it.

    Args:
        label:
            The label of the car or the group of cars.

        x:
            The initial position, one value per car or one for all.

        values:
            Further values by name, such as the starting speed `"x'"`.

        size:
            The number of cars, or `None` for one.

        space:
            The road, a continuous space of one dimension whose axis `x` makes the position of a car its place, or
            `None` for none.

    Examples:
        >>> from skcomplex.systems.engineering.common.traffic import CarElement
        >>> CarElement('cars', x=[0.0, 10.0], values={"x'": 1.0}, size=2).x
        [0.0, 10.0]
    """

    def __init__(
        self,
        label: str,
        x: object = 0.0,
        *,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.x = x
