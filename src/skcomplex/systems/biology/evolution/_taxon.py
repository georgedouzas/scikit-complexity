"""Species of an ecosystem with its field, the sum of the couplings of the species it depends on, and its age."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class TaxonElement(Element):
    """Species of an ecosystem with its field, the sum of the couplings of the species it depends on, and its age.

    A species whose field is negative goes extinct, and a new species takes its niche.

    Args:
        label:
            The label of the species or of the group of species.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        h:
            The field, the sum of the couplings of its dependencies.

        age:
            The number of steps since the species appeared.

    Examples:
        >>> from skcomplex.systems.biology.evolution import TaxonElement
        >>> TaxonElement('species').h
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        h: Annotated[float, StateSpec()] = 1.0,
        age: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.h = h
        self.age = age
