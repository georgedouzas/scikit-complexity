"""Define population densities, the density of a structured population at one class."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.spaces import BaseSpace
from ....general.elements import NonnegativeElement


class PopulationDensityElement(NonnegativeElement):
    """Value `x` of the density of a structured population at one class.

    A structured population is this element on a lattice of classes, such as sizes or activities, with one value per
    class, and growth, encounters, and transport change it. The element has no laws of its own.

    Args:
        label:
            The label of the element, unique among the components of its system.

        x:
            The initial density, a nonnegative number.

        size:
            The number of members, or `None` for one.

        space:
            The space with one member per site, such as a lattice of classes, or `None` for none.

    Examples:
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> from skcomplex.systems.biology.common.populations import PopulationDensityElement
        >>> classes = LatticeSpace(5, boundary='fixed')
        >>> PopulationDensityElement('cells', x=[0.2] * 5, space=classes).x
        [0.2, 0.2, 0.2, 0.2, 0.2]
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, x=x, size=size, space=space)
