"""Define points, members of a continuous space whose vector state has order 2."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt

from ....core.components.elements import Element
from ....core.components.spaces import BaseSpace
from ....core.specs import StateSpec, VisualSpec


class PointElement(Element):
    """Point of a continuous space.

    The point is a member with a vector state `x` of order 2 that locates it. The laws that act on the point give the
    second derivative of `x`, and the companion state `x_d1` holds its first derivative. The boundary of the space acts
    on `x`.

    Args:
        label:
            The label of the point.

        x:
            The initial value of the vector state. A number applies to every component.

        x_d1:
            The initial value of the first derivative of the vector state. A number applies to every component.

        visual:
            How the point is drawn, or `None` for `default_visual_`.

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

    def __init__(
        self,
        label: str,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, order=2, role='position')] = 0.0,
        x_d1: npt.ArrayLike = 0.0,
        visual: VisualSpec | None = None,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set `default_visual_`, points at the values of `x`.

        Args:
            context:
                The space the element inherits from its system, which the element does not read.
        """
        super()._prepare(context)
        self.default_visual_ = VisualSpec('points', {'position': 'x'})
