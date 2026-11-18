"""Node of a network with triadic regulation.

The node is active when it belongs to the giant component of the active links.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from .....core.components.elements import Element
from .....core.components.spaces import BaseSpace
from .....core.specs import VisualSpec

ACTIVITY_VISUAL = VisualSpec('sites', {'states': ['active'], 'low': 0.0, 'high': 1.0})


class RegulatedNodeElement(Element):
    """Node of a network with triadic regulation.

    The node is active when it belongs to the giant component of the active links.

    Args:
        label:
            The label of the node group.

        active:
            1 when the node is in the giant component of the active links, and 0 otherwise.

        space:
            The network whose nodes hold one member each, or `None` for one node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import RegulatedNodeElement
        >>> RegulatedNodeElement('nodes').active
        1.0
    """

    visual = ACTIVITY_VISUAL

    def __init__(
        self,
        label: str,
        *,
        active: Annotated[object, FiniteEnumeratedSet([0, 1])] = 1.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.active = active
