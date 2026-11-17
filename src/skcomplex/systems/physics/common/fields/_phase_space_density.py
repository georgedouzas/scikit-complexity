"""Define phase-space densities, the one-particle density of kinetic theory at the points of a phase space."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.elements import Element
from .....core.components.spaces import BaseSpace


class PhaseSpaceDensityElement(Element):
    """Value `x` of the one-particle density of kinetic theory at every point of a phase space.

    A gas is one member per position and velocity of a `PhaseSpace`, and streaming, collisions, and relaxation change
    it. The density is nonnegative.

    Args:
        label:
            The label of the density, unique among the components of its system.

        x:
            The initial density, one nonnegative value per point or one for all.

        space:
            The phase space of the points, or `None` for one point.

    Examples:
        >>> from skcomplex.core.components.spaces import AxisSpec, LatticeSpace, PhaseSpace
        >>> from skcomplex.systems.physics.common.fields import PhaseSpaceDensityElement
        >>> space = PhaseSpace(LatticeSpace(4), [AxisSpec('v', 8, -3.0, 3.0)])
        >>> PhaseSpaceDensityElement('gas', x=0.2, space=space).space.prepare().size_
        32
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.x = x
