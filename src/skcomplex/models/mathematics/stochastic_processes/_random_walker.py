"""Walker on the integers that steps at the rate `rate`, up with the probability `bias`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class RandomWalkerElement(Element):
    """Walker on the integers that steps at the rate `rate`, up with the probability `bias`.

    Args:
        label:
            The label of the walker or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        rate:
            The rate of steps.

        bias:
            The probability that a step goes up.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import RandomWalkerElement
        >>> RandomWalkerElement('walker', bias=0.7).bias
        0.7
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=ZZ, role='position')] = 0.0,
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        bias: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.rate = rate
        self.bias = bias
