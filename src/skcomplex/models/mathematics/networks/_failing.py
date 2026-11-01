"""Node of a network under failures or attacks, with its degree and the parameters of its removal."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._vertex import DEGREE_VISUAL


class FailingElement(Element):
    """Node of a network under failures or attacks, with its degree and the parameters of its removal.

    The removal of the model reads the parameters: every step a node fails with probability `rate` under random
    failures, and under a targeted attack the nodes of the largest degree are removed, one degree class per step.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        degree:
            The number of edges of the node, read at the start of every step.

        rate:
            The probability that a node fails in a step under random failures.

        targeted:
            1 for a targeted attack on the nodes of the largest degree and 0 for random failures.

    Examples:
        >>> from skcomplex.models.mathematics.networks import FailingElement
        >>> FailingElement('nodes', targeted=1).targeted
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = DEGREE_VISUAL,
        *,
        degree: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        targeted: Annotated[float, ParameterSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.degree = degree
        self.rate = rate

        self.targeted = targeted
