"""Define population densities, the density of a structured population at one class."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.elements import NonnegativeElement
from .....core.specs import StateSpec, VisualSpec


class PopulationDensityElement(NonnegativeElement):
    """Value `x` of the density of a structured population at one class.

    A structured population is a group of these elements on the sites of a lattice of classes, such as sizes or
    activities, one member per class, and growth, encounters, and transport change it. The element has no laws of its
    own.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial density, a nonnegative number.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> from skcomplex.systems.biology.common.populations import PopulationDensityElement
        >>> classes = LatticeSpace(5, boundary='fixed')
        >>> values = {'x': 0.2}
        >>> group = GroupElement('cells', element=PopulationDensityElement, space=classes, on='sites', values=values)
        >>> group.prepare().size_
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, x=x)
