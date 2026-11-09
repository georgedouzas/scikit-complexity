"""Define phase-space densities, the one-particle density of kinetic theory at one point."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.specs import StateSpec, VisualSpec
from ....general.elements import NonnegativeElement


class PhaseSpaceDensityElement(NonnegativeElement):
    """Value `x` of the one-particle density of kinetic theory at one point of phase space.

    A gas is a field of these elements on a `PhaseSpace`, one member per position and velocity, and streaming,
    collisions, and relaxation change it. The element has no laws of its own.

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
        >>> from skcomplex.systems.physics.common.fields import PhaseSpaceDensityElement
        >>> classes = LatticeSpace(5, boundary='fixed')
        >>> values = {'x': 0.2}
        >>> group = GroupElement('gas', element=PhaseSpaceDensityElement, space=classes, on='sites', values=values)
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
