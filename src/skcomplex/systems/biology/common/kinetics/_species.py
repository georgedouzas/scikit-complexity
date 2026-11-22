"""Define species, elements that hold the concentration of one molecular species."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.spaces import BaseSpace
from ....general.elements import NonnegativeElement


class SpeciesElement(NonnegativeElement):
    """Molecular species with a nonnegative concentration `x`.

    It is an mRNA, an enzyme, a messenger, or a metabolite.

    A species has no laws of its own, and reactions synthesize, convert, and degrade it.

    Args:
        label:
            The label of the species.

        x:
            The initial concentration.

        size:
            The number of members, or `None` for one.

        space:
            The space with one member per site, such as a lattice of classes, or `None` for none.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import SpeciesElement
        >>> SpeciesElement('camp', x=2.0).x
        2.0
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
