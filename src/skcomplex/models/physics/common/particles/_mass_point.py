"""Define point masses, points of a continuous space with a mass."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.elements import PointElement
from .....specs import ParameterSpec, VisualSpec


class MassPointElement(PointElement):
    """Point mass, a point whose position `x` has order 2, with its velocity `x_d1` and a mass `m`.

    The forces on the point divide by the mass through their divisor `'m'`.

    Args:
        label:
            The label of the point.

        x:
            The initial position. A number applies to every component.

        x_d1:
            The initial velocity. A number applies to every component.

        visual:
            How the point is drawn, or `None` for `default_visual_`.

        m:
            The mass.

    Attributes:
        default_visual_:
            Points at the positions.

    Examples:
        >>> from skcomplex.models.physics.common.particles import MassPointElement
        >>> MassPointElement('ball', x=[0.0, 1.0], m=2.0).m
        2.0
    """

    def __init__(
        self,
        label: str,
        x: npt.ArrayLike = 0.0,
        x_d1: npt.ArrayLike = 0.0,
        visual: VisualSpec | None = None,
        *,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, x=x, x_d1=x_d1, visual=visual)
        self.m = m
