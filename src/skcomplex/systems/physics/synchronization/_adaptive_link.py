"""Directed link of an adaptive network with the coupling weight `kappa`, a state that the phases of its ends adapt."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class AdaptiveLinkElement(Element):
    """Directed link of an adaptive network with the coupling weight `kappa`, a state that the phases of its ends adapt.

    Args:
        label:
            The label of the link or the group on the links.

        visual:
            How the link is drawn, or `None` for `default_visual_`.

        kappa:
            The initial coupling weight.

    Attributes:
        default_visual_:
            The visual of the link when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import AdaptiveLinkElement
        >>> AdaptiveLinkElement('link', kappa=0.5).kappa
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        kappa: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.kappa = kappa
