"""Species of a microbial community, with an abundance and an intrinsic growth rate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class MicrobeElement(Element):
    """Species of a microbial community, with an abundance and an intrinsic growth rate.

    Args:
        label:
            The label of the species or of the group of species.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial abundance.

        b:
            The intrinsic growth rate.

    Examples:
        >>> from skcomplex.systems.biology.ecology import MicrobeElement
        >>> MicrobeElement('species', x=0.1, b=0.8).b
        0.8
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        b: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.b = b
