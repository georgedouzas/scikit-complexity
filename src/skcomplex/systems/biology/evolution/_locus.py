"""Locus of the genome shared by a monomorphic population, with an allele of `-1` or `1`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import ZZ

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class LocusElement(Element):
    """Locus of the genome shared by a monomorphic population, with an allele of `-1` or `1`.

    Args:
        label:
            The label of the locus or the group of loci.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        s:
            The allele, `-1` or `1`.

        h:
            The random field of the locus, the fitness gained by the allele `1` over the allele `-1`, halved.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.evolution import LocusElement
        >>> GroupElement('genome', element=LocusElement, size=10, values={'h': 0.05}).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('image', {'low': -1.0, 'high': 1.0, 'history': 64}),
        *,
        s: Annotated[float, StateSpec(domain=ZZ)] = -1.0,
        h: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.s = s
        self.h = h
