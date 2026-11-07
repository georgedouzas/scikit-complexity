"""Node of two interdependent networks, functional while it belongs to the giant component of both."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec
from ._regulated_node import ACTIVITY_VISUAL


class DependentNodeElement(Element):
    """Node of two interdependent networks, functional while it belongs to the giant component of both.

    The node stands for a pair of nodes, one of each network, that depend on each other, so it fails when either of
    them leaves the giant component of the functional nodes of its network.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        active:
            Whether the node is functional, 1 or 0.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import DependentNodeElement
        >>> DependentNodeElement('nodes', active=0.0).active
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = ACTIVITY_VISUAL,
        *,
        active: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.active = active
