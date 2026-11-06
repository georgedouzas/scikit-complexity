"""Share of a population whose genomes lie a given number of point mutations away from the master sequence."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class GenotypeClassElement(Element):
    """Share of a population whose genomes lie a given number of point mutations away from the master sequence.

    Args:
        label:
            The label of the class.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial share.

    Examples:
        >>> from skcomplex.systems.biology.evolution import GenotypeClassElement
        >>> GenotypeClassElement('k1', x=0.1).x
        0.1
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
