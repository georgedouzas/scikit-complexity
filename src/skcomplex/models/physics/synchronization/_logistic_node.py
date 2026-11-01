"""Logistic map on a node of a network, with its value, growth rate, and weight of the neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class LogisticNodeElement(Element):
    """Logistic map on a node of a network, with its value, growth rate, and weight of the neighbours.

    `LogisticCouplingInteraction` iterates `x_i(t + 1) = (1 - epsilon) f(x_i) + epsilon mean_j f(x_j)`, where the map
    `f(x) = r x (1 - x)` of the node mixes with the mean of the maps of its neighbours `j` on the network.

    Args:
        label:
            The label of the node or the group on the nodes.

        visual:
            How the node is drawn, or `None` for `default_visual_`.

        x:
            The initial value.

        r:
            The growth rate of the logistic map.

        epsilon:
            The weight of the neighbours.

    Attributes:
        default_visual_:
            The visual of the node when it sets none.

    Examples:
        >>> from skcomplex.models.physics.synchronization import LogisticNodeElement
        >>> LogisticNodeElement('node').epsilon
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.5,
        r: Annotated[float, ParameterSpec()] = 4.0,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.r = r
        self.epsilon = epsilon
