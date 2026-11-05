"""Define densities, elements that hold the value of a density at one class."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ...specs import StateSpec, VisualSpec
from ._element import Element


class DensityElement(Element):
    """Value of a density at one class, a member of a distribution over the classes.

    A distribution is a group of densities on the sites of a lattice of classes, one member per class, and encounter,
    proliferation, and destruction laws change it. A density has no laws of its own.

    Args:
        label:
            The label of the density, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        f:
            The initial density, a nonnegative number.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        >>> from skcomplex.core.elements import DensityElement, GroupElement
        >>> from skcomplex.core.spaces import LatticeSpace
        >>> classes = LatticeSpace(5, boundary='fixed')
        >>> group = GroupElement('distribution', element=DensityElement, space=classes, on='sites', values={'f': 0.2})
        >>> group.prepare().size_
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        f: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.f = f
