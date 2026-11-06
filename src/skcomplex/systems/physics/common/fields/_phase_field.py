"""Define phase fields, the sites of a field of a scalar order parameter."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from .....core.components.elements import Element
from .....core.specs import StateSpec, VisualSpec


class PhaseFieldElement(Element):
    """Site of a phase field, which holds the order parameter `phi` that free energies and noise move.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        phi:
            The initial order parameter.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.common.fields import PhaseFieldElement
        >>> PhaseFieldElement('site', phi=0.5).phi
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi
