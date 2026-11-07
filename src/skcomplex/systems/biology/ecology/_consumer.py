"""Consumer species that grows on the energy it takes from resources and needs a maintenance energy to persist."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ConsumerElement(Element):
    """Consumer species that grows on the energy it takes from resources and needs a maintenance energy to persist.

    Args:
        label:
            The label of the species or of the group of species.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        N:
            The initial population size.

        e:
            The efficiency that converts the energy taken in into growth.

        m:
            The maintenance energy, the intake at which the population neither grows nor declines.

    Examples:
        >>> from skcomplex.systems.biology.ecology import ConsumerElement
        >>> ConsumerElement('consumers', m=0.5).m
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        N: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        e: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.N = N
        self.e = e
        self.m = m
