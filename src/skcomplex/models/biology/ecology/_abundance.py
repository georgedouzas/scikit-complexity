"""Species of an ecosystem with its relative abundance and its age, which goes extinct below a threshold."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class AbundanceElement(Element):
    """Species of an ecosystem with its relative abundance and its age, which goes extinct below a threshold.

    Args:
        label:
            The label of the species or of the group of species.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The relative abundance.

        age:
            The number of steps since the species appeared.

        threshold:
            The abundance below which the species goes extinct.

    Examples:
        >>> from skcomplex.models.biology.ecology import AbundanceElement
        >>> AbundanceElement('species').threshold
        0.001
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        age: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        threshold: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.age = age
        self.threshold = threshold
