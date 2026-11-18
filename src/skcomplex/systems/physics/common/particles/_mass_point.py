"""Define point masses, points of a continuous space with a mass."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.spaces import BaseSpace
from ....general.elements import PointElement


class MassPointElement(PointElement):
    """Point mass, a point at the position `x` with the mass `m`.

    The forces on the point give the acceleration `x''` of its position divided by its mass, and its velocity is the
    value `x'`. It is an atom of a gas, a bead of a polymer, or a star of a cluster.

    Args:
        label:
            The label of the point.

        x:
            The initial position. A number applies to every component.

        m:
            The mass.

        values:
            Further values by name, such as the starting velocity `"x'"`.

        size:
            The number of points, or `None` for one.

        space:
            The space whose sites hold one point each, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.common.particles import MassPointElement
        >>> MassPointElement('ball', x=[0.0, 1.0], m=2.0, values={"x'": [1.0, 0.0]}).m
        2.0
    """

    def __init__(
        self,
        label: str,
        x: npt.ArrayLike = 0.0,
        *,
        m: Annotated[float, RealSet.open(0, oo)] = 1.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, x=x, values=values, size=size, space=space)
        self.m = m
