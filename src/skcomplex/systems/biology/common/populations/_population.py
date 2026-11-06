"""Define populations, compartments that hold the size of a population of one species."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.elements import CompartmentElement
from .....core.specs import StateSpec, VisualSpec


class PopulationElement(CompartmentElement):
    """Population of one species, a compartment whose state `n` holds its size.

    Args:
        label:
            The label of the population.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial size.

    Examples:
        >>> from skcomplex.systems.biology.common.populations import PopulationElement
        >>> PopulationElement('rabbits', n=40).n
        40
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual, n=n)
