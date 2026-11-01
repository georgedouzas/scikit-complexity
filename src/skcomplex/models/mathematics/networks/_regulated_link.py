"""Link of a network with triadic regulation, with its activity and the sign of its regulating node."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class RegulatedLinkElement(Element):
    """Link of a network with triadic regulation, with its activity and the sign of its regulating node.

    Args:
        label:
            The label of the link group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        active:
            1 when the link is active, and 0 otherwise.

        sign:
            1 when the regulating node lets the link be active while the node is active, and -1 when it lets it be
            active while the node is inactive.

    Examples:
        >>> from skcomplex.models.mathematics.networks import RegulatedLinkElement
        >>> RegulatedLinkElement('links', sign=-1).sign
        -1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        active: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
        sign: Annotated[float, ParameterSpec(domain=FiniteEnumeratedSet([-1, 1]))] = 1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.active = active
        self.sign = sign
