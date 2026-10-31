"""Define species, elements that hold the concentration of one molecular species."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec


class SpeciesElement(Element):
    """Molecular species, an mRNA, an enzyme, a messenger, or a metabolite, with a nonnegative concentration `x`.

    A species has no laws of its own, and reactions synthesize, convert, and degrade it.

    Args:
        label:
            The label of the species.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial concentration.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import SpeciesElement
        >>> SpeciesElement('camp', x=2.0).x
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
