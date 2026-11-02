"""Node of a network with triadic regulation, active when it belongs to the giant component of the active links."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec

ACTIVITY_VISUAL = VisualSpec('sites', {'states': ['active'], 'low': 0.0, 'high': 1.0})


class RegulatedNodeElement(Element):
    """Node of a network with triadic regulation, active when it belongs to the giant component of the active links.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        active:
            1 when the node is in the giant component of the active links, and 0 otherwise.

    Examples:
        >>> from skcomplex.models.mathematics.networks import RegulatedNodeElement
        >>> RegulatedNodeElement('nodes').active
        1.0
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
