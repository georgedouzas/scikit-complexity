"""Define compartments, elements that hold the amount of units in one named state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.spaces import BaseSpace
from ....general.elements import NonnegativeElement


class CompartmentElement(NonnegativeElement):
    """Compartment, the amount `x` of units in one named state, as a count or a fraction.

    The label of the element names the compartment, such as `'S'`, `'I'`, or `'plasma'`. A compartment has no laws of
    its own, and the reactions and flows of epidemics, pharmacokinetics, and populations move its units.

    Args:
        label:
            The label of the compartment, unique among the components of its system.

        x:
            The initial amount, a nonnegative count or fraction.

        size:
            The number of members, or `None` for one.

        space:
            The space with one member per site, such as a lattice of classes, or `None` for none.

    Examples:
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.biology.common.kinetics import CompartmentElement
        >>> system = System(elements=[CompartmentElement('S', x=990), CompartmentElement('I', x=10)]).prepare()
        >>> system.parameters_
        ('S.x', 'I.x')
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
