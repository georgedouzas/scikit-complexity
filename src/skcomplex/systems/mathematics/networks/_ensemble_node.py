"""Node of a random graph, with its degree and the probability that it adds an edge to a random node per step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._vertex import DEGREE_VISUAL


class EnsembleNodeElement(Element):
    """Node of a random graph, with its degree and the probability that it adds an edge to a random node per step.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        degree:
            The number of edges of the node, read at the start of every step.

        rate:
            The probability that the node adds an edge to a random node in a step.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import EnsembleNodeElement
        >>> EnsembleNodeElement('nodes', rate=0.1).rate
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = DEGREE_VISUAL,
        *,
        degree: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.degree = degree
        self.rate = rate
